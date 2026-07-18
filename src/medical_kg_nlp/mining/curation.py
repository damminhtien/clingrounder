"""Policy-driven annotation partitioning for model-specific training views."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from medical_kg_nlp.mining.records import AnnotationProposal, ReviewStatus

__all__ = [
    "AnnotationCurationPolicy",
    "AnnotationCurationResult",
    "curate_annotations",
    "load_annotation_curation_policy",
]


@dataclass(frozen=True)
class AnnotationCurationPolicy:
    """Reusable eligibility rules for one derived annotation training view."""

    policy_id: str
    allowed_review_statuses: frozenset[ReviewStatus]
    allow_discontinuous: bool = True
    reject_import_issues: bool = True
    max_span_length: int | None = None

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.allowed_review_statuses:
            raise ValueError("Annotation curation policy must have an ID and statuses")
        if self.max_span_length is not None and self.max_span_length <= 0:
            raise ValueError("max_span_length must be positive when configured")


@dataclass(frozen=True)
class AnnotationCurationResult:
    """Accepted and rejected source records plus deterministic reason counts."""

    accepted: tuple[AnnotationProposal, ...]
    rejected: tuple[AnnotationProposal, ...]
    report: dict[str, Any]


def load_annotation_curation_policy(path: str | Path) -> AnnotationCurationPolicy:
    """Load a strict policy so training eligibility is versioned and auditable."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Annotation curation policy must be an object")
    if raw.get("schema_version") != "medical-annotation-curation-policy.v1":
        raise ValueError("Unsupported annotation curation policy schema version")
    raw_statuses = raw.get("allowed_review_statuses")
    if not isinstance(raw_statuses, list) or not raw_statuses:
        raise ValueError("allowed_review_statuses must be a non-empty list")
    maximum = raw.get("max_span_length")
    return AnnotationCurationPolicy(
        policy_id=str(raw["policy_id"]),
        allowed_review_statuses=frozenset(ReviewStatus(str(value)) for value in raw_statuses),
        allow_discontinuous=_boolean(raw, "allow_discontinuous", default=True),
        reject_import_issues=_boolean(raw, "reject_import_issues", default=True),
        max_span_length=None if maximum is None else int(maximum),
    )


def curate_annotations(
    annotations: Sequence[AnnotationProposal],
    policy: AnnotationCurationPolicy,
) -> AnnotationCurationResult:
    """Partition without rewriting source labels, spans, concepts, or provenance."""

    annotation_ids = [annotation.annotation_id for annotation in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Cannot curate annotations with duplicate IDs")
    accepted = []
    rejected = []
    reason_counts: Counter[str] = Counter()
    rejected_by_reason: dict[str, list[str]] = {}
    for annotation in sorted(annotations, key=lambda item: item.annotation_id):
        reasons = _rejection_reasons(annotation, policy)
        if not reasons:
            accepted.append(annotation)
            continue
        rejected.append(annotation)
        for reason in reasons:
            reason_counts[reason] += 1
        rejected_by_reason[annotation.annotation_id] = reasons
    accepted_types = Counter(annotation.entity_type for annotation in accepted)
    rejected_types = Counter(annotation.entity_type for annotation in rejected)
    report = {
        "schema_version": "medical-annotation-curation.v1",
        "policy_id": policy.policy_id,
        "input_count": len(annotations),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_entity_types": dict(sorted(accepted_types.items())),
        "rejected_entity_types": dict(sorted(rejected_types.items())),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        # Rejected source records remain unchanged in JSONL. This compact index
        # explains routing decisions without duplicating mutable annotation copies.
        "rejected_annotation_reasons": dict(sorted(rejected_by_reason.items())),
        "policy": {
            "allowed_review_statuses": sorted(
                status.value for status in policy.allowed_review_statuses
            ),
            "allow_discontinuous": policy.allow_discontinuous,
            "reject_import_issues": policy.reject_import_issues,
            "max_span_length": policy.max_span_length,
        },
    }
    return AnnotationCurationResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        report=report,
    )


def _rejection_reasons(
    annotation: AnnotationProposal,
    policy: AnnotationCurationPolicy,
) -> list[str]:
    reasons = []
    if annotation.review_status not in policy.allowed_review_statuses:
        reasons.append(f"review_status:{annotation.review_status.value}")
    if not policy.allow_discontinuous and annotation.metadata.get("discontinuous") == "true":
        reasons.append("discontinuous")
    if policy.reject_import_issues and annotation.metadata.get("import_issues", "[]") not in {
        "",
        "[]",
    }:
        reasons.append("import_issue")
    span_length = annotation.span[1] - annotation.span[0]
    if policy.max_span_length is not None and span_length > policy.max_span_length:
        reasons.append("span_too_long")
    return reasons


def _boolean(raw: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value
