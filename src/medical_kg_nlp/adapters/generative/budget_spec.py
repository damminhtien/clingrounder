"""Reproducible inference-budget specifications with artifact-level evidence."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from medical_kg_nlp.adapters.generative.budget import (
    InferenceBudgetManifest,
    ModelBudgetEntry,
)
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.io import read_yaml

__all__ = [
    "BudgetReservation",
    "InferenceBudgetSpec",
    "ModelParameterEvidence",
    "load_inference_budget_spec",
    "safetensors_parameter_count",
    "verify_inference_budget_spec",
]

EvidenceKind = Literal["manifest", "safetensors"]


@dataclass(frozen=True, slots=True)
class ModelParameterEvidence:
    """Pinned bytes that prove one active artifact's declared parameter count."""

    kind: EvidenceKind
    path: Path
    sha256: str
    parameter_field: str | None = None

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("Parameter evidence requires a lowercase SHA-256")
        if self.kind == "manifest" and not self.parameter_field:
            raise ValueError("Manifest evidence requires parameter_field")
        if self.kind == "safetensors" and self.parameter_field is not None:
            raise ValueError("Safetensors evidence cannot define parameter_field")

    def measured_parameter_count(self) -> int:
        """Verify immutable bytes and return their recorded tensor count."""

        if not self.path.is_file():
            raise ValueError(f"Parameter evidence is absent: {self.path}")
        actual_sha256 = sha256_file(self.path)
        if actual_sha256 != self.sha256:
            raise ValueError(
                "Parameter evidence SHA-256 mismatch: "
                f"expected {self.sha256}, got {actual_sha256}"
            )
        if self.kind == "safetensors":
            return safetensors_parameter_count(self.path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        value: object = payload
        assert self.parameter_field is not None
        for component in self.parameter_field.split("."):
            if not isinstance(value, dict) or component not in value:
                raise ValueError(
                    f"Parameter field {self.parameter_field!r} is absent from "
                    f"{self.path}"
                )
            value = value[component]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"Parameter field {self.parameter_field!r} must be a positive integer"
            )
        return value


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    """Upper bound for a learned artifact that has not been promoted yet."""

    artifact_id: str
    maximum_parameters: int
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("Budget reservation artifact_id must be non-empty")
        if self.maximum_parameters <= 0:
            raise ValueError("Budget reservation maximum_parameters must be positive")
        expected_roles = tuple(sorted(set(role.strip() for role in self.roles if role.strip())))
        if not expected_roles or self.roles != expected_roles:
            raise ValueError("Budget reservation roles must be non-empty, unique, and sorted")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable reservation record without inventing model provenance."""

        return {
            "artifact_id": self.artifact_id,
            "maximum_parameters": self.maximum_parameters,
            "roles": list(self.roles),
            "status": "reserved",
        }


@dataclass(frozen=True, slots=True)
class InferenceBudgetSpec:
    """Active learned artifacts plus fail-closed capacity for future artifacts."""

    schema_version: str
    config_path: Path
    run_root: Path
    active: InferenceBudgetManifest
    evidence: tuple[ModelParameterEvidence, ...]
    reservations: tuple[BudgetReservation, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "inference-model-budget-spec.v1":
            raise ValueError("Unsupported inference budget spec schema")
        if len(self.active.entries) != len(self.evidence):
            raise ValueError("Every active artifact requires one parameter-count evidence")
        active_ids = {entry.artifact_id for entry in self.active.entries}
        reserved_ids = {entry.artifact_id for entry in self.reservations}
        if len(reserved_ids) != len(self.reservations):
            raise ValueError("Budget reservation artifact IDs must be unique")
        if active_ids & reserved_ids:
            raise ValueError("An artifact cannot be both active and reserved")
        if self.total_parameters > self.active.maximum_parameters:
            raise ValueError(
                "Inference model budget plus reservations exceeded: "
                f"{self.total_parameters:,} > {self.active.maximum_parameters:,}"
            )

    @property
    def reserved_parameters(self) -> int:
        """Return the maximum parameter capacity held for untrained artifacts."""

        return sum(item.maximum_parameters for item in self.reservations)

    @property
    def total_parameters(self) -> int:
        """Return active measured counts plus all fail-closed reservations."""

        return self.active.total_parameters + self.reserved_parameters

    def relative_path(self, path: Path) -> str:
        """Project a verified artifact path into the portable run root."""

        try:
            return path.resolve().relative_to(self.run_root).as_posix()
        except ValueError as error:
            raise ValueError(f"Budget evidence path escapes run_root: {path}") from error


def load_inference_budget_spec(path: str | Path) -> InferenceBudgetSpec:
    """Load a strict budget plan without importing Torch or Transformers."""

    config_path = Path(path).resolve()
    raw = read_yaml(config_path)
    run_root = _resolve(config_path.parent, _required_string(raw, "run_root"))
    try:
        config_path.relative_to(run_root)
    except ValueError as error:
        raise ValueError("Inference budget config must live below run_root") from error
    maximum_parameters = int(raw.get("maximum_parameters", 9_000_000_000))
    active_raw = _list_of_mappings(raw.get("active"), "active")
    entries: list[ModelBudgetEntry] = []
    evidence: list[ModelParameterEvidence] = []
    for item in active_raw:
        roles = _string_tuple(item.get("roles"), "active roles")
        entries.append(
            ModelBudgetEntry(
                artifact_id=_required_string(item, "artifact_id"),
                model_id=_required_string(item, "model_id"),
                revision=_required_string(item, "revision"),
                parameter_count=int(item["parameter_count"]),
                kind=cast(Any, str(item.get("kind", "auxiliary"))),
                roles=roles,
            )
        )
        evidence_raw = _mapping(item.get("evidence"), "active evidence")
        evidence_path = _resolve(run_root, _required_string(evidence_raw, "path"))
        _relative_to_root(evidence_path, run_root)
        evidence.append(
            ModelParameterEvidence(
                kind=cast(EvidenceKind, _required_string(evidence_raw, "kind")),
                path=evidence_path,
                sha256=_required_string(evidence_raw, "sha256"),
                parameter_field=_optional_string(evidence_raw.get("parameter_field")),
            )
        )
    reservations = tuple(
        BudgetReservation(
            artifact_id=_required_string(item, "artifact_id"),
            maximum_parameters=int(item["maximum_parameters"]),
            roles=_string_tuple(item.get("roles"), "reservation roles"),
        )
        for item in _list_of_mappings(raw.get("reservations", []), "reservations")
    )
    return InferenceBudgetSpec(
        schema_version=_required_string(raw, "schema_version"),
        config_path=config_path,
        run_root=run_root,
        active=InferenceBudgetManifest(
            entries=tuple(entries),
            maximum_parameters=maximum_parameters,
        ),
        evidence=tuple(evidence),
        reservations=reservations,
    )


def verify_inference_budget_spec(spec: InferenceBudgetSpec) -> dict[str, Any]:
    """Verify every active count and return the portable budget manifest."""

    active_records: list[dict[str, Any]] = []
    for entry, evidence in zip(spec.active.entries, spec.evidence, strict=True):
        measured = evidence.measured_parameter_count()
        if measured != entry.parameter_count:
            raise ValueError(
                f"Parameter count mismatch for {entry.artifact_id}: "
                f"expected {entry.parameter_count:,}, got {measured:,}"
            )
        active_records.append(
            {
                **entry.to_dict(),
                "evidence": {
                    "kind": evidence.kind,
                    "path": spec.relative_path(evidence.path),
                    "sha256": evidence.sha256,
                    **(
                        {"parameter_field": evidence.parameter_field}
                        if evidence.parameter_field is not None
                        else {}
                    ),
                },
                "measured_parameter_count": measured,
                "status": "active",
            }
        )
    return {
        "schema_version": "inference-model-budget.v2",
        "status": "verified",
        "config": {
            "path": spec.relative_path(spec.config_path),
            "sha256": sha256_file(spec.config_path),
        },
        "maximum_parameters": spec.active.maximum_parameters,
        "active_parameters": spec.active.total_parameters,
        "reserved_parameters": spec.reserved_parameters,
        "total_parameters": spec.total_parameters,
        "remaining_parameters": spec.active.maximum_parameters - spec.total_parameters,
        "active": active_records,
        "reservations": [item.to_dict() for item in spec.reservations],
    }


def safetensors_parameter_count(path: str | Path) -> int:
    """Count tensors from a Safetensors header without loading model weights."""

    tensor_path = Path(path)
    file_size = tensor_path.stat().st_size
    with tensor_path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Safetensors header is truncated: {tensor_path}")
        (header_length,) = struct.unpack("<Q", raw_length)
        # INVARIANT: reject corrupt lengths before allocating memory from untrusted artifacts.
        if header_length <= 0 or header_length > file_size - 8:
            raise ValueError(f"Safetensors header length is invalid: {tensor_path}")
        header = json.loads(handle.read(header_length).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header must be an object: {tensor_path}")
    total = 0
    for name, raw_tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(raw_tensor, dict):
            raise ValueError(f"Safetensors metadata for {name!r} must be an object")
        shape = raw_tensor.get("shape")
        offsets = raw_tensor.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and value >= 0 for value in offsets)
            or offsets[0] > offsets[1]
        ):
            raise ValueError(f"Invalid Safetensors metadata for {name!r}")
        total += math.prod(shape)
    if total <= 0:
        raise ValueError(f"Safetensors artifact contains no parameters: {tensor_path}")
    return total


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _relative_to_root(path: Path, run_root: Path) -> None:
    try:
        path.relative_to(run_root)
    except ValueError as error:
        raise ValueError(f"Budget evidence path escapes run_root: {path}") from error


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _list_of_mappings(value: object, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [_mapping(item, field) for item in value]


def _required_string(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Optional string must be non-empty when present")
    return value.strip()


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return tuple(sorted(str(item).strip() for item in value))
