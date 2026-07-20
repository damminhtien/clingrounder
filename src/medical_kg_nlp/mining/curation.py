"""Policy-driven annotation partitioning for model-specific training views."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    ReviewStatus,
)

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
    allowed_layers: frozenset[AnnotationLayer] | None = None
    allowed_entity_types: frozenset[str] = frozenset()
    allowed_metadata_values: dict[str, frozenset[str]] = field(default_factory=dict)
    overlap_strategy: Literal["preserve", "prefer_quality_longest"] = "preserve"

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.allowed_review_statuses:
            raise ValueError("Annotation curation policy must have an ID and statuses")
        if self.max_span_length is not None and self.max_span_length <= 0:
            raise ValueError("max_span_length must be positive when configured")
        if self.allowed_layers is not None and not self.allowed_layers:
            raise ValueError("allowed_layers cannot be empty when configured")
        if any(not entity_type.strip() for entity_type in self.allowed_entity_types):
            raise ValueError("allowed_entity_types must contain non-empty values")
        for key, values in self.allowed_metadata_values.items():
            if not key.strip() or not values or any(not value.strip() for value in values):
                raise ValueError(
                    "allowed_metadata_values requires non-empty keys and value sets"
                )


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
    raw_layers = raw.get("allowed_layers")
    if raw_layers is not None and not isinstance(raw_layers, list):
        raise ValueError("allowed_layers must be a list when configured")
    raw_entity_types = raw.get("allowed_entity_types", [])
    if not isinstance(raw_entity_types, list):
        raise ValueError("allowed_entity_types must be a list")
    raw_metadata_values = raw.get("allowed_metadata_values", {})
    if not isinstance(raw_metadata_values, Mapping):
        raise ValueError("allowed_metadata_values must be an object")
    metadata_values: dict[str, frozenset[str]] = {}
    for key, values in raw_metadata_values.items():
        if not isinstance(values, list):
            raise ValueError("allowed_metadata_values entries must be lists")
        metadata_values[str(key)] = frozenset(str(value) for value in values)
    overlap_strategy = str(raw.get("overlap_strategy", "preserve"))
    if overlap_strategy not in {"preserve", "prefer_quality_longest"}:
        raise ValueError("Unsupported overlap_strategy")
    return AnnotationCurationPolicy(
        policy_id=str(raw["policy_id"]),
        allowed_review_statuses=frozenset(ReviewStatus(str(value)) for value in raw_statuses),
        allow_discontinuous=_boolean(raw, "allow_discontinuous", default=True),
        reject_import_issues=_boolean(raw, "reject_import_issues", default=True),
        max_span_length=None if maximum is None else int(maximum),
        allowed_layers=(
            None
            if raw_layers is None
            else frozenset(AnnotationLayer(str(value)) for value in raw_layers)
        ),
        allowed_entity_types=frozenset(str(value) for value in raw_entity_types),
        allowed_metadata_values=metadata_values,
        overlap_strategy=cast(
            Literal["preserve", "prefer_quality_longest"], overlap_strategy
        ),
    )


def curate_annotations(
    annotations: Sequence[AnnotationProposal],
    policy: AnnotationCurationPolicy,
) -> AnnotationCurationResult:
    """Partition without rewriting source labels, spans, concepts, or provenance."""

    annotation_ids = [annotation.annotation_id for annotation in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Cannot curate annotations with duplicate IDs")
    eligible: list[AnnotationProposal] = []
    rejected: list[AnnotationProposal] = []
    reason_counts: Counter[str] = Counter()
    rejected_by_reason: dict[str, list[str]] = {}
    for annotation in sorted(annotations, key=lambda item: item.annotation_id):
        reasons = _rejection_reasons(annotation, policy)
        if not reasons:
            eligible.append(annotation)
            continue
        rejected.append(annotation)
        for reason in reasons:
            reason_counts[reason] += 1
        rejected_by_reason[annotation.annotation_id] = reasons
    overlap_winners: dict[str, str] = {}
    if policy.overlap_strategy == "prefer_quality_longest":
        accepted, overlap_rejected, overlap_winners = _resolve_overlaps(eligible)
        rejected.extend(overlap_rejected)
        for annotation in overlap_rejected:
            reason_counts["overlap_lower_priority"] += 1
            rejected_by_reason[annotation.annotation_id] = ["overlap_lower_priority"]
    else:
        accepted = eligible
    accepted.sort(key=lambda item: item.annotation_id)
    rejected.sort(key=lambda item: item.annotation_id)
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
            "allowed_layers": (
                None
                if policy.allowed_layers is None
                else sorted(layer.value for layer in policy.allowed_layers)
            ),
            "allowed_entity_types": sorted(policy.allowed_entity_types),
            "allowed_metadata_values": {
                key: sorted(values)
                for key, values in sorted(policy.allowed_metadata_values.items())
            },
            "overlap_strategy": policy.overlap_strategy,
        },
        "overlap_winners": dict(sorted(overlap_winners.items())),
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
    if policy.allowed_layers is not None and annotation.layer not in policy.allowed_layers:
        reasons.append(f"layer:{annotation.layer.value}")
    if (
        policy.allowed_entity_types
        and annotation.entity_type not in policy.allowed_entity_types
    ):
        reasons.append(f"entity_type:{annotation.entity_type}")
    for key, allowed_values in policy.allowed_metadata_values.items():
        if annotation.metadata.get(key) not in allowed_values:
            reasons.append(f"metadata:{key}")
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


def _resolve_overlaps(
    annotations: Sequence[AnnotationProposal],
) -> tuple[list[AnnotationProposal], list[AnnotationProposal], dict[str, str]]:
    by_document: dict[str, list[AnnotationProposal]] = defaultdict(list)
    for annotation in annotations:
        by_document[annotation.document_id].append(annotation)
    accepted: list[AnnotationProposal] = []
    rejected: list[AnnotationProposal] = []
    winners: dict[str, str] = {}
    for document_id in sorted(by_document):
        selected: list[AnnotationProposal] = []
        for annotation in sorted(by_document[document_id], key=_overlap_priority):
            winner = next(
                (candidate for candidate in selected if _overlaps(annotation, candidate)),
                None,
            )
            if winner is None:
                selected.append(annotation)
            else:
                rejected.append(annotation)
                winners[annotation.annotation_id] = winner.annotation_id
        accepted.extend(selected)
    return accepted, rejected, winners


def _overlap_priority(annotation: AnnotationProposal) -> tuple[int, int, float, int, int, str]:
    review_rank = {
        ReviewStatus.ACCEPTED: 0,
        ReviewStatus.PROPOSED: 1,
        ReviewStatus.NEEDS_REVIEW: 2,
        ReviewStatus.REJECTED: 3,
    }[annotation.review_status]
    layer_rank = {
        AnnotationLayer.GOLD: 0,
        AnnotationLayer.CHALLENGE: 0,
        AnnotationLayer.SILVER: 1,
        AnnotationLayer.BRONZE: 2,
    }[annotation.layer]
    span_length = annotation.span[1] - annotation.span[0]
    return (
        review_rank,
        layer_rank,
        -annotation.confidence,
        -span_length,
        annotation.span[0],
        annotation.annotation_id,
    )


def _overlaps(left: AnnotationProposal, right: AnnotationProposal) -> bool:
    return left.span[0] < right.span[1] and right.span[0] < left.span[1]


def _boolean(raw: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value
