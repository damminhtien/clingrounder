"""Portable reviewer-agreement artifacts for public dataset release gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

__all__ = ["ReviewAgreementArtifact"]

_SCHEMA = "clingrounder.review-agreement.v1"


@dataclass(frozen=True, slots=True)
class ReviewAgreementArtifact:
    """A text-free, dataset-bound summary of independent reviewer agreement."""

    dataset_id: str
    dataset_version: str
    reviewed_document_count: int
    double_reviewed_document_count: int
    double_review_fraction: float
    span_type_agreement: float | None
    assertion_agreement: float | None
    relation_agreement: float | None
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        for name, value in (("dataset_id", self.dataset_id), ("dataset_version", self.dataset_version)):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.schema_version != _SCHEMA:
            raise ValueError(f"unsupported review agreement schema: {self.schema_version}")
        if self.reviewed_document_count < 1:
            raise ValueError("reviewed_document_count must be positive")
        if not 1 <= self.double_reviewed_document_count <= self.reviewed_document_count:
            raise ValueError("double_reviewed_document_count must be within reviewed documents")
        if not _unit_interval(self.double_review_fraction):
            raise ValueError("double_review_fraction must be finite and in [0, 1]")
        expected_fraction = self.double_reviewed_document_count / self.reviewed_document_count
        if not math.isclose(self.double_review_fraction, expected_fraction, abs_tol=1e-9):
            raise ValueError("double_review_fraction does not match reviewer counts")
        for name in ("span_type_agreement", "assertion_agreement", "relation_agreement"):
            value = getattr(self, name)
            if value is not None and not _unit_interval(value):
                raise ValueError(f"{name} must be null or a finite value in [0, 1]")

    @classmethod
    def from_report(
        cls,
        dataset_id: str,
        dataset_version: str,
        report: Mapping[str, Any],
    ) -> "ReviewAgreementArtifact":
        """Convert the mining quality report without carrying reviewer identities or text."""

        return cls(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            reviewed_document_count=_required_int(report, "reviewed_document_count"),
            double_reviewed_document_count=_required_int(report, "double_reviewed_document_count"),
            double_review_fraction=_required_float(report, "double_review_fraction"),
            span_type_agreement=_optional_float(report, "span_type_agreement"),
            assertion_agreement=_optional_float(report, "assertion_agreement"),
            relation_agreement=_optional_float(report, "relation_agreement"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render deterministic JSON suitable for a manifest-pinned release artifact."""

        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "reviewed_document_count": self.reviewed_document_count,
            "double_reviewed_document_count": self.double_reviewed_document_count,
            "double_review_fraction": self.double_review_fraction,
            "span_type_agreement": self.span_type_agreement,
            "assertion_agreement": self.assertion_agreement,
            "relation_agreement": self.relation_agreement,
        }

    def write(self, path: str | Path) -> None:
        """Write the artifact with stable formatting and no raw review content."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"review agreement field {key!r} must be an integer")
    return value


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    value = _optional_float(payload, key)
    if value is None:
        raise ValueError(f"review agreement field {key!r} must be numeric")
    return value


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"review agreement field {key!r} must be numeric or null")
    return float(value)


def _unit_interval(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )
