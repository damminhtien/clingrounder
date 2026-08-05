"""Explicit data retention and artifact loading policy.

These controls describe runtime behavior. They are not a regulatory certification.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

__all__ = ["DataPolicy", "GovernancePolicy"]


@dataclass(frozen=True)
class DataPolicy:
    """PHI-minimizing defaults for logs, traces, and document metadata."""

    logging_level: str = "WARNING"
    text_retention: str = "none"
    trace_retention: str = "memory_only"
    hash_document_ids: bool = True
    metadata_allowlist: tuple[str, ...] = ()
    deletion_behavior: str = "best_effort_unlink"

    def __post_init__(self) -> None:
        if self.logging_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("logging_level must be a standard Python logging level")
        if self.text_retention not in {"none", "memory", "explicit_file"}:
            raise ValueError("text_retention must be none, memory, or explicit_file")
        if self.trace_retention not in {"none", "memory_only", "explicit_file"}:
            raise ValueError("trace_retention must be none, memory_only, or explicit_file")
        if self.deletion_behavior not in {"best_effort_unlink", "retain_until_explicit_delete"}:
            raise ValueError("unsupported deletion_behavior")


@dataclass(frozen=True)
class GovernancePolicy:
    """Runtime governance configuration with fail-closed local artifact defaults."""

    data: DataPolicy = DataPolicy()
    allowed_artifact_roots: tuple[str, ...] = ()
    artifact_allowlist: tuple[tuple[str, str], ...] = ()
    local_files_only: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "GovernancePolicy":
        allowed = {
            "data",
            "allowed_artifact_roots",
            "artifact_allowlist",
            "local_files_only",
        }
        unknown = sorted(str(key) for key in payload if str(key) not in allowed)
        if unknown:
            raise ValueError(f"Unknown governance config keys: {', '.join(unknown)}")
        data_payload = payload.get("data", {})
        if not isinstance(data_payload, Mapping):
            raise ValueError("governance.data must be a mapping")
        data_allowed = {
            "logging_level",
            "text_retention",
            "trace_retention",
            "hash_document_ids",
            "metadata_allowlist",
            "deletion_behavior",
        }
        data_unknown = sorted(str(key) for key in data_payload if str(key) not in data_allowed)
        if data_unknown:
            raise ValueError(f"Unknown governance.data config keys: {', '.join(data_unknown)}")
        roots = _sequence(payload.get("allowed_artifact_roots", ()), "allowed_artifact_roots")
        allowlist = _sequence(payload.get("artifact_allowlist", ()), "artifact_allowlist")
        return cls(
            data=DataPolicy(
                logging_level=str(data_payload.get("logging_level", "WARNING")),
                text_retention=str(data_payload.get("text_retention", "none")),
                trace_retention=str(data_payload.get("trace_retention", "memory_only")),
                hash_document_ids=bool(data_payload.get("hash_document_ids", True)),
                metadata_allowlist=_string_tuple(
                    data_payload.get("metadata_allowlist", ()), "metadata_allowlist"
                ),
                deletion_behavior=str(data_payload.get("deletion_behavior", "best_effort_unlink")),
            ),
            allowed_artifact_roots=tuple(str(value) for value in roots),
            artifact_allowlist=tuple(
                _allowlist_item(item) for item in allowlist
            ),
            local_files_only=bool(payload.get("local_files_only", True)),
        )


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"governance.{name} must be a sequence")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value, name))


def _allowlist_item(value: object) -> tuple[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError("governance.artifact_allowlist items require path and sha256")
    return str(value["path"]), str(value["sha256"])
