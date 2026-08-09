"""Portable manifests and verification for released local model artifacts.

The runtime must be able to identify a model before loading it.  This module deliberately
does not import a model framework: a manifest can be inspected and verified on a clean CPU
machine before optional Transformers or PyTorch dependencies are installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from clingrounder.governance.artifacts import verify_artifact
from clingrounder.governance.models import ModelGovernanceMetadata

__all__ = [
    "MODEL_ARTIFACT_SCHEMA_VERSION",
    "ModelArtifactManifest",
    "load_model_artifact_manifest",
    "verify_model_artifact",
]

MODEL_ARTIFACT_SCHEMA_VERSION = "clingrounder.model-artifact.v1"


@dataclass(frozen=True, slots=True)
class ModelArtifactManifest:
    """Identity, provenance, and evaluation contract for one model directory or file."""

    artifact_id: str
    task: str
    model_id: str
    revision: str
    artifact_sha256: str
    governance: ModelGovernanceMetadata
    training_data_fingerprints: tuple[str, ...]
    config_sha256: str
    # Research artifacts may omit these until their release evidence is complete.  An approved
    # artifact must pin both the tokenizer and the runtime profile before it can be loaded under
    # the release policy.
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    runtime_profile: str | None = None
    metrics: tuple[tuple[str, float], ...] = ()
    schema_version: str = MODEL_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "artifact_id",
            "task",
            "model_id",
            "revision",
            "artifact_sha256",
            "config_sha256",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.schema_version != MODEL_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported model artifact schema: {self.schema_version}")
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        _validate_sha256(self.config_sha256, "config_sha256")
        if not self.training_data_fingerprints:
            raise ValueError("training_data_fingerprints must not be empty")
        for fingerprint in self.training_data_fingerprints:
            _validate_sha256(fingerprint, "training_data_fingerprint")
        if (self.tokenizer_id is None) != (self.tokenizer_revision is None):
            raise ValueError("tokenizer_id and tokenizer_revision must be supplied together")
        if self.tokenizer_id is not None:
            if not self.tokenizer_id.strip():
                raise ValueError("tokenizer_id must be non-empty")
            _validate_revision(self.tokenizer_revision, "tokenizer_revision")
        if self.runtime_profile is not None and not self.runtime_profile.strip():
            raise ValueError("runtime_profile must be non-empty when supplied")
        seen_metrics: set[str] = set()
        for name, value in self.metrics:
            if not name.strip() or name in seen_metrics:
                raise ValueError("metrics must contain unique, non-empty names")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"metric {name!r} must be between 0 and 1")
            seen_metrics.add(name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize deterministically without loading the model framework."""

        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "task": self.task,
            "model": {
                "model_id": self.model_id,
                "revision": self.revision,
                "artifact_sha256": self.artifact_sha256,
            },
            "training_data_fingerprints": list(self.training_data_fingerprints),
            "config_sha256": self.config_sha256,
            "tokenizer": (
                None
                if self.tokenizer_id is None
                else {"id": self.tokenizer_id, "revision": self.tokenizer_revision}
            ),
            "runtime_profile": self.runtime_profile,
            "metrics": dict(self.metrics),
            "governance": {
                "model_id": self.governance.model_id,
                "revision": self.governance.revision,
                "training_data_description": self.governance.training_data_description,
                "intended_use": self.governance.intended_use,
                "excluded_use": self.governance.excluded_use,
                "evaluation_summary": self.governance.evaluation_summary,
                "known_limitations": self.governance.known_limitations,
                "approval_status": self.governance.approval_status,
                "rollback_model": self.governance.rollback_model,
            },
        }

    def write(self, path: str | Path) -> None:
        """Write a stable JSON manifest; callers own atomic publication of the directory."""

        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelArtifactManifest":
        """Parse a manifest while rejecting unknown shape and missing provenance fields."""

        _reject_unknown(
            payload,
            {
                "schema_version",
                "artifact_id",
                "task",
                "model",
                "training_data_fingerprints",
                "config_sha256",
                "tokenizer",
                "runtime_profile",
                "metrics",
                "governance",
            },
            "model artifact",
        )
        if payload.get("schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported model artifact schema")
        model = _mapping(payload, "model")
        _reject_unknown(
            model,
            {"model_id", "revision", "artifact_sha256"},
            "model",
        )
        governance_raw = _mapping(payload, "governance")
        _reject_unknown(
            governance_raw,
            {
                "model_id",
                "revision",
                "training_data_description",
                "intended_use",
                "excluded_use",
                "evaluation_summary",
                "known_limitations",
                "approval_status",
                "rollback_model",
            },
            "governance",
        )
        tokenizer_raw = payload.get("tokenizer")
        tokenizer_id: str | None = None
        tokenizer_revision: str | None = None
        if tokenizer_raw is not None:
            tokenizer = _mapping_value(tokenizer_raw, "tokenizer")
            _reject_unknown(tokenizer, {"id", "revision"}, "tokenizer")
            tokenizer_id = _string(tokenizer, "id")
            tokenizer_revision = _string(tokenizer, "revision")
        runtime_profile = payload.get("runtime_profile")
        if runtime_profile is not None and (
            not isinstance(runtime_profile, str) or not runtime_profile.strip()
        ):
            raise ValueError("runtime_profile must be a non-empty string or null")
        metrics_raw = payload.get("metrics", {})
        if not isinstance(metrics_raw, Mapping):
            raise ValueError("metrics must be an object")
        governance = ModelGovernanceMetadata(
            model_id=_string(governance_raw, "model_id"),
            revision=_string(governance_raw, "revision"),
            training_data_description=_string(governance_raw, "training_data_description"),
            intended_use=_string(governance_raw, "intended_use"),
            excluded_use=_string(governance_raw, "excluded_use"),
            evaluation_summary=_string(governance_raw, "evaluation_summary"),
            known_limitations=_string(governance_raw, "known_limitations"),
            approval_status=str(governance_raw.get("approval_status", "unreviewed")),
            rollback_model=governance_raw.get("rollback_model"),
        )
        fingerprints = payload.get("training_data_fingerprints")
        if not isinstance(fingerprints, list) or not all(isinstance(item, str) for item in fingerprints):
            raise ValueError("training_data_fingerprints must be a list of strings")
        return cls(
            artifact_id=_string(payload, "artifact_id"),
            task=_string(payload, "task"),
            model_id=_string(model, "model_id"),
            revision=_string(model, "revision"),
            artifact_sha256=_string(model, "artifact_sha256"),
            governance=governance,
            training_data_fingerprints=tuple(fingerprints),
            config_sha256=_string(payload, "config_sha256"),
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            runtime_profile=runtime_profile,
            metrics=tuple((str(name), float(value)) for name, value in metrics_raw.items()),
        )


def load_model_artifact_manifest(path: str | Path) -> ModelArtifactManifest:
    """Load one JSON manifest without importing optional ML dependencies."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read model artifact manifest: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("model artifact manifest must be a JSON object")
    return ModelArtifactManifest.from_mapping(payload)


def verify_model_artifact(
    artifact_path: str | Path,
    manifest: ModelArtifactManifest,
    *,
    require_approved: bool = False,
) -> str:
    """Verify artifact identity and return its SHA-256 before model initialization.

    ``require_approved`` is intended for a release/application gate.  Research users may
    inspect an unreviewed artifact, but the distinction remains explicit in the manifest.
    """

    if require_approved and manifest.governance.approval_status != "approved":
        raise ValueError("model artifact is not approved for this load policy")
    if require_approved and (
        manifest.tokenizer_id is None
        or manifest.tokenizer_revision is None
        or manifest.runtime_profile is None
    ):
        raise ValueError(
            "approved model artifacts must pin tokenizer and runtime_profile provenance"
        )
    if manifest.governance.model_id != manifest.model_id:
        raise ValueError("governance model_id does not match artifact model_id")
    if manifest.governance.revision != manifest.revision:
        raise ValueError("governance revision does not match artifact revision")
    return verify_artifact(artifact_path, manifest.artifact_sha256)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _mapping_value(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"unknown {field_name} fields: {', '.join(unknown)}")


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _validate_sha256(value: str, field_name: str) -> None:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field_name} must be a SHA-256 digest")


def _validate_revision(value: str | None, field_name: str) -> None:
    if value is None or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{field_name} must be a 40-character lowercase commit SHA")
