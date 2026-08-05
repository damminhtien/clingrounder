"""Human-readable governance metadata for model approval and rollback."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelGovernanceMetadata"]


@dataclass(frozen=True)
class ModelGovernanceMetadata:
    model_id: str
    revision: str
    training_data_description: str
    intended_use: str
    excluded_use: str
    evaluation_summary: str
    known_limitations: str
    approval_status: str = "unreviewed"
    rollback_model: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "model_id",
            "revision",
            "training_data_description",
            "intended_use",
            "excluded_use",
            "evaluation_summary",
            "known_limitations",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.approval_status not in {"unreviewed", "approved", "rejected", "deprecated"}:
            raise ValueError("unsupported model approval_status")
