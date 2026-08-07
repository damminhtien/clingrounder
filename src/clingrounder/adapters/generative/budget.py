"""Parameter-budget validation for reproducible model-assisted inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["InferenceBudgetManifest", "ModelBudgetEntry"]

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
ModelArtifactKind = Literal["base", "adapter", "auxiliary"]


@dataclass(frozen=True, slots=True)
class ModelBudgetEntry:
    """One distinct learned artifact loaded by an inference pipeline.

    Reusing the same checkpoint for multiple passes does not create another entry. A LoRA adapter
    is a separate learned artifact and therefore contributes its own trainable parameter count.
    Quantization is intentionally absent because it changes memory, not the number of parameters.
    """

    artifact_id: str
    model_id: str
    revision: str
    parameter_count: int
    kind: ModelArtifactKind
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.artifact_id.strip() or not self.model_id.strip():
            raise ValueError("Model budget entries require artifact_id and model_id")
        if self.kind not in {"base", "adapter", "auxiliary"}:
            raise ValueError("Model kind must be base, adapter, or auxiliary")
        # MODEL: mutable branches cannot prove which checkpoint produced a submission.
        if _COMMIT_SHA.fullmatch(self.revision) is None:
            raise ValueError("Model revision must be a full 40-character commit SHA")
        if self.parameter_count <= 0:
            raise ValueError("Model parameter_count must be positive")
        expected_roles = tuple(sorted(set(role.strip() for role in self.roles if role.strip())))
        if not expected_roles or self.roles != expected_roles:
            raise ValueError("Model roles must be non-empty, unique, and sorted")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, machine-readable budget record."""

        return {
            "artifact_id": self.artifact_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "parameter_count": self.parameter_count,
            "kind": self.kind,
            "roles": list(self.roles),
        }


@dataclass(frozen=True, slots=True)
class InferenceBudgetManifest:
    """Validate the sum of distinct learned artifacts used for one output."""

    entries: tuple[ModelBudgetEntry, ...]
    maximum_parameters: int = 9_000_000_000

    def __post_init__(self) -> None:
        if self.maximum_parameters <= 0:
            raise ValueError("maximum_parameters must be positive")
        artifact_ids = [entry.artifact_id for entry in self.entries]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("Each learned artifact must appear exactly once in a model budget")
        identities = [(entry.model_id, entry.revision, entry.kind) for entry in self.entries]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "Repeated use of one checkpoint must be represented by multiple roles, "
                "not duplicate budget entries"
            )
        if self.total_parameters > self.maximum_parameters:
            raise ValueError(
                "Inference model budget exceeded: "
                f"{self.total_parameters:,} > {self.maximum_parameters:,}"
            )

    @property
    def total_parameters(self) -> int:
        """Return the parameter count before any memory-only quantization."""

        return sum(entry.parameter_count for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the budget for experiment and submission manifests."""

        return {
            "schema_version": "inference-model-budget.v1",
            "maximum_parameters": self.maximum_parameters,
            "total_parameters": self.total_parameters,
            "remaining_parameters": self.maximum_parameters - self.total_parameters,
            "entries": [entry.to_dict() for entry in self.entries],
        }
