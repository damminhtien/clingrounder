"""Inter-reviewer agreement metrics and release gates for mined gold data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RelationProposal,
    ReviewStatus,
)

__all__ = [
    "AgreementThresholds",
    "GoldAgreementGate",
    "ReviewAgreementEvaluator",
    "ReviewAgreementReport",
]


@dataclass(frozen=True)
class AgreementThresholds:
    """Minimum quality targets defined by the mining campaign."""

    span_type: float = 0.90
    assertion: float = 0.85
    relation: float = 0.80
    double_review_fraction: float = 0.10

    def __post_init__(self) -> None:
        for name, value in (
            ("span_type", self.span_type),
            ("assertion", self.assertion),
            ("relation", self.relation),
            ("double_review_fraction", self.double_review_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} threshold must be in [0, 1]")


@dataclass(frozen=True)
class ReviewAgreementReport:
    """Pairwise exact-set agreement over independently reviewed documents."""

    reviewed_document_count: int
    double_reviewed_document_count: int
    double_review_fraction: float
    span_type_agreement: float | None
    assertion_agreement: float | None
    relation_agreement: float | None
    entity_pair_count: int
    relation_pair_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewed_document_count": self.reviewed_document_count,
            "double_reviewed_document_count": self.double_reviewed_document_count,
            "double_review_fraction": self.double_review_fraction,
            "span_type_agreement": self.span_type_agreement,
            "assertion_agreement": self.assertion_agreement,
            "relation_agreement": self.relation_agreement,
            "entity_pair_count": self.entity_pair_count,
            "relation_pair_count": self.relation_pair_count,
        }


class ReviewAgreementEvaluator:
    """Compute reviewer agreement without depending on a task label schema."""

    def evaluate(
        self,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
        relations: Sequence[RelationProposal] = (),
    ) -> ReviewAgreementReport:
        document_ids = {document.document_id for document in documents}
        accepted_annotations = [
            annotation
            for annotation in annotations
            if annotation.document_id in document_ids
            and annotation.layer in {AnnotationLayer.GOLD, AnnotationLayer.CHALLENGE}
            and annotation.review_status is ReviewStatus.ACCEPTED
        ]
        by_document_reviewer: dict[
            tuple[str, str], list[AnnotationProposal]
        ] = defaultdict(list)
        reviewers_by_document: dict[str, set[str]] = defaultdict(set)
        for annotation in accepted_annotations:
            by_document_reviewer[(annotation.document_id, annotation.labeler_id)].append(
                annotation
            )
            reviewers_by_document[annotation.document_id].add(annotation.labeler_id)

        accepted_relations = [
            relation
            for relation in relations
            if relation.document_id in document_ids
            and relation.layer in {AnnotationLayer.GOLD, AnnotationLayer.CHALLENGE}
            and relation.review_status is ReviewStatus.ACCEPTED
            and relation.labeler_id is not None
        ]
        relations_by_document_reviewer: dict[
            tuple[str, str], list[RelationProposal]
        ] = defaultdict(list)
        for relation in accepted_relations:
            assert relation.labeler_id is not None
            relations_by_document_reviewer[
                (relation.document_id, relation.labeler_id)
            ].append(relation)

        span_scores: list[float] = []
        assertion_scores: list[float] = []
        relation_scores: list[float] = []
        annotations_by_id = {
            annotation.annotation_id: annotation for annotation in accepted_annotations
        }
        for document_id, reviewers in sorted(reviewers_by_document.items()):
            for left_reviewer, right_reviewer in combinations(sorted(reviewers), 2):
                left = by_document_reviewer[(document_id, left_reviewer)]
                right = by_document_reviewer[(document_id, right_reviewer)]
                span_scores.append(_jaccard(_entity_keys(left), _entity_keys(right)))
                assertion_scores.append(
                    _jaccard(_assertion_keys(left), _assertion_keys(right))
                )
                left_relations = _relation_keys(
                    relations_by_document_reviewer.get(
                        (document_id, left_reviewer), []
                    ),
                    annotations_by_id,
                )
                right_relations = _relation_keys(
                    relations_by_document_reviewer.get(
                        (document_id, right_reviewer), []
                    ),
                    annotations_by_id,
                )
                if left_relations or right_relations:
                    relation_scores.append(_jaccard(left_relations, right_relations))

        reviewed_count = len(reviewers_by_document)
        double_reviewed_count = sum(
            len(reviewers) >= 2 for reviewers in reviewers_by_document.values()
        )
        return ReviewAgreementReport(
            reviewed_document_count=reviewed_count,
            double_reviewed_document_count=double_reviewed_count,
            double_review_fraction=(
                double_reviewed_count / reviewed_count if reviewed_count else 0.0
            ),
            span_type_agreement=_mean_or_none(span_scores),
            assertion_agreement=_mean_or_none(assertion_scores),
            relation_agreement=_mean_or_none(relation_scores),
            entity_pair_count=len(span_scores),
            relation_pair_count=len(relation_scores),
        )


class GoldAgreementGate:
    """Promote campaign quality targets to blocking snapshot issues."""

    def __init__(self, thresholds: AgreementThresholds | None = None) -> None:
        self.thresholds = thresholds or AgreementThresholds()

    def validate(
        self,
        report: ReviewAgreementReport,
        *,
        has_gold_relations: bool,
    ) -> tuple[str, ...]:
        if report.reviewed_document_count == 0:
            return ()
        issues: list[str] = []
        if report.double_review_fraction < self.thresholds.double_review_fraction:
            issues.append(
                "double_review_fraction:"
                f"{report.double_review_fraction:.6f}<"
                f"{self.thresholds.double_review_fraction:.6f}"
            )
        _append_agreement_issue(
            issues,
            "span_type_agreement",
            report.span_type_agreement,
            self.thresholds.span_type,
        )
        _append_agreement_issue(
            issues,
            "assertion_agreement",
            report.assertion_agreement,
            self.thresholds.assertion,
        )
        if has_gold_relations:
            _append_agreement_issue(
                issues,
                "relation_agreement",
                report.relation_agreement,
                self.thresholds.relation,
            )
        return tuple(issues)


def _entity_keys(values: Sequence[AnnotationProposal]) -> set[tuple[tuple[int, int], str]]:
    return {(value.span, value.entity_type) for value in values}


def _assertion_keys(
    values: Sequence[AnnotationProposal],
) -> set[tuple[tuple[int, int], str, tuple[str, ...]]]:
    return {
        (value.span, value.entity_type, tuple(sorted(value.assertions))) for value in values
    }


def _relation_keys(
    values: Sequence[RelationProposal],
    annotations: Mapping[str, AnnotationProposal],
) -> set[tuple[tuple[int, int], str, tuple[int, int], str, str]]:
    result = set()
    for relation in values:
        head = annotations.get(relation.head_annotation_id)
        tail = annotations.get(relation.tail_annotation_id)
        if head is None or tail is None:
            continue
        result.add(
            (
                head.span,
                head.entity_type,
                tail.span,
                tail.entity_type,
                relation.relation_type,
            )
        )
    return result


def _jaccard(left: Set[object], right: Set[object]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _append_agreement_issue(
    issues: list[str], name: str, value: float | None, threshold: float
) -> None:
    if value is None:
        issues.append(f"{name}:unmeasured")
    elif value < threshold:
        issues.append(f"{name}:{value:.6f}<{threshold:.6f}")
