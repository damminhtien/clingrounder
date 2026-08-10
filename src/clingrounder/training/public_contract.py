"""Framework-neutral contract for a future public Vietnamese NER model.

This contract is intentionally separate from competition run specs.  It makes the release
boundary executable without requiring a GPU, while ``pending_public_snapshot`` prevents an
unfinished data release from being mistaken for a trainable public model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping

import yaml

__all__ = [
    "PublicTrainingContract",
    "load_public_training_contract",
]

_SCHEMA = "clingrounder.public-training-contract.v1"
_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_STATUS = Literal["pending_public_snapshot", "ready"]
_ROOT_FIELDS = {
    "schema_version",
    "run_id",
    "status",
    "task",
    "labels",
    "model",
    "data",
    "training",
    "selection",
}
_MODEL_FIELDS = {"id", "revision"}
_DATA_FIELDS = {
    "train_manifest",
    "validation_manifest",
    "train_manifest_sha256",
    "validation_manifest_sha256",
    "dataset_audit_report",
    "dataset_audit_report_sha256",
}
_TRAINING_FIELDS = {"seed", "epochs", "batch_size", "learning_rate", "weight_decay"}
_SELECTION_FIELDS = {"primary_metric", "minimum_primary_metric"}


@dataclass(frozen=True, slots=True)
class PublicTrainingContract:
    """Pinned model/data/training policy used before producing a release artifact."""

    config_path: Path
    run_id: str
    status: _STATUS
    task: str
    labels: tuple[str, ...]
    model_id: str
    model_revision: str
    train_manifest: Path | None
    validation_manifest: Path | None
    train_manifest_sha256: str | None
    validation_manifest_sha256: str | None
    dataset_audit_report: Path | None
    dataset_audit_report_sha256: str | None
    seed: int
    epochs: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    primary_metric: str
    minimum_primary_metric: float
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA:
            raise ValueError(f"unsupported public training schema: {self.schema_version}")
        for name in ("run_id", "task", "model_id", "model_revision", "primary_metric"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.status not in {"pending_public_snapshot", "ready"}:
            raise ValueError(f"unsupported public training status: {self.status}")
        if _SHA.fullmatch(self.model_revision) is None:
            raise ValueError("model_revision must be a 40-character lowercase commit SHA")
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be non-empty and unique")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.epochs <= 0 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.minimum_primary_metric <= 1.0:
            raise ValueError("minimum_primary_metric must be between 0 and 1")
        if self.status == "ready" and (
            self.train_manifest is None
            or self.validation_manifest is None
            or self.train_manifest_sha256 is None
            or self.validation_manifest_sha256 is None
            or self.dataset_audit_report is None
            or self.dataset_audit_report_sha256 is None
        ):
            raise ValueError(
                "ready training contracts require manifests, SHA-256 values, and an audit report"
            )
        if self.train_manifest is not None and self.validation_manifest is not None:
            if self.train_manifest == self.validation_manifest:
                raise ValueError("train and validation manifests must differ")
        for name, path, fingerprint in (
            ("train_manifest", self.train_manifest, self.train_manifest_sha256),
            ("validation_manifest", self.validation_manifest, self.validation_manifest_sha256),
            ("dataset_audit_report", self.dataset_audit_report, self.dataset_audit_report_sha256),
        ):
            if path is None and fingerprint is not None:
                raise ValueError(f"{name} SHA-256 requires a manifest path")
            if fingerprint is not None:
                _validate_sha256(fingerprint, f"{name}_sha256")
                if not path or sha256_file(path) != fingerprint:
                    raise ValueError(f"{name} SHA-256 mismatch")
        if self.status == "ready":
            assert self.dataset_audit_report is not None
            try:
                audit = json.loads(self.dataset_audit_report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("ready training contract audit report is unreadable") from error
            if not isinstance(audit, Mapping) or audit.get("eligible_for_clinical_claim") is not True:
                raise ValueError("ready training contract requires an eligible dataset audit report")

    def to_dict(self, *, root: Path | None = None) -> dict[str, Any]:
        """Render stable, portable configuration for a run manifest."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status,
            "task": self.task,
            "labels": list(self.labels),
            "model": {"id": self.model_id, "revision": self.model_revision},
            "data": {
                "train_manifest": _portable_path(self.train_manifest, root),
                "validation_manifest": _portable_path(self.validation_manifest, root),
                "train_manifest_sha256": self.train_manifest_sha256,
                "validation_manifest_sha256": self.validation_manifest_sha256,
                "dataset_audit_report": _portable_path(self.dataset_audit_report, root),
                "dataset_audit_report_sha256": self.dataset_audit_report_sha256,
            },
            "training": {
                "seed": self.seed,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
            "selection": {
                "primary_metric": self.primary_metric,
                "minimum_primary_metric": self.minimum_primary_metric,
            },
        }


def load_public_training_contract(path: str | Path) -> PublicTrainingContract:
    """Load and validate a public contract without importing optional ML dependencies."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read public training contract: {config_path}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("public training contract must be a YAML object")
    schema_version = raw.get("schema_version")
    if schema_version != _SCHEMA:
        raise ValueError(f"schema_version: unsupported public training schema: {schema_version!r}")
    _reject_unknown(raw, _ROOT_FIELDS, "")
    model = _mapping(raw, "model", "model")
    data = _mapping(raw, "data", "data")
    training = _mapping(raw, "training", "training")
    selection = _mapping(raw, "selection", "selection")
    _reject_unknown(model, _MODEL_FIELDS, "model")
    _reject_unknown(data, _DATA_FIELDS, "data")
    _reject_unknown(training, _TRAINING_FIELDS, "training")
    _reject_unknown(selection, _SELECTION_FIELDS, "selection")
    status = raw.get("status")
    if status not in {"pending_public_snapshot", "ready"}:
        raise ValueError("status must be pending_public_snapshot or ready")
    labels = raw.get("labels")
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ValueError("labels must be a list of strings")
    return PublicTrainingContract(
        config_path=config_path,
        run_id=_string(raw, "run_id"),
        status=status,
        task=_string(raw, "task"),
        labels=tuple(labels),
        model_id=_string(model, "id"),
        model_revision=_string(model, "revision"),
        train_manifest=_resolve_optional_path(data.get("train_manifest"), config_path.parent),
        validation_manifest=_resolve_optional_path(
            data.get("validation_manifest"), config_path.parent
        ),
        train_manifest_sha256=_optional_sha256(data.get("train_manifest_sha256"), "train_manifest_sha256"),
        validation_manifest_sha256=_optional_sha256(
            data.get("validation_manifest_sha256"), "validation_manifest_sha256"
        ),
        dataset_audit_report=_resolve_optional_path(
            data.get("dataset_audit_report"), config_path.parent
        ),
        dataset_audit_report_sha256=_optional_sha256(
            data.get("dataset_audit_report_sha256"), "dataset_audit_report_sha256"
        ),
        seed=_integer(training, "seed"),
        epochs=_number(training, "epochs"),
        batch_size=_integer(training, "batch_size"),
        learning_rate=_number(training, "learning_rate"),
        weight_decay=_number(training, "weight_decay"),
        primary_metric=_string(selection, "primary_metric"),
        minimum_primary_metric=_number(selection, "minimum_primary_metric"),
        schema_version=schema_version,
    )


def _mapping(payload: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], path: str) -> None:
    """Fail closed so a typo cannot silently turn into a defaulted training option."""

    for key in payload:
        if key not in allowed:
            field_path = f"{path}.{key}" if path else str(key)
            raise ValueError(f"unknown field: {field_path}")


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _optional_sha256(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 digest or null")
    return value


def _resolve_optional_path(value: Any, root: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("dataset manifest paths must be strings or null")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"training manifest escapes contract directory: {value}") from error
    return path


def _portable_path(path: Path | None, root: Path | None) -> str | None:
    if path is None:
        return None
    if root is None:
        return str(path)
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    """Hash a contract input in chunks so large dataset manifests do not fill memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, field_name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 digest")
