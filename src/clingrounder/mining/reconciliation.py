"""Exact-text duplicate reconciliation with annotation disagreement preservation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

from clingrounder.mining.policy import MiningQualityGate
from clingrounder.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)

__all__ = [
    "DocumentCanonicalMapping",
    "DuplicateReconciliationReport",
    "ExactDuplicateReconciliationResult",
    "reconcile_exact_duplicates",
]

_SemanticKey = tuple[
    tuple[int, int],
    str,
    str,
    tuple[str, ...],
    tuple[tuple[str, str, str], ...],
    str,
    str,
]


@dataclass(frozen=True)
class DocumentCanonicalMapping:
    """Map one source document to the retained exact-text representative."""

    document_id: str
    canonical_document_id: str
    text_sha256: str
    group_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "canonical_document_id": self.canonical_document_id,
            "text_sha256": self.text_sha256,
            "group_size": self.group_size,
        }


@dataclass(frozen=True)
class DuplicateReconciliationReport:
    """Agreement and materialization counts for one reconciliation run."""

    input_document_count: int
    input_annotation_count: int
    output_document_count: int
    output_training_annotation_count: int
    review_annotation_count: int
    duplicate_group_count: int
    duplicate_document_count: int
    agreement_pair_count: int
    exact_micro_jaccard: float | None
    exact_macro_jaccard: float | None
    exact_agreed_key_count: int
    exact_union_key_count: int
    source_label_agreement: tuple[tuple[str, int, int, float], ...]
    groups: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "medical-duplicate-reconciliation.v1",
            "input_document_count": self.input_document_count,
            "input_annotation_count": self.input_annotation_count,
            "output_document_count": self.output_document_count,
            "output_training_annotation_count": self.output_training_annotation_count,
            "review_annotation_count": self.review_annotation_count,
            "duplicate_group_count": self.duplicate_group_count,
            "duplicate_document_count": self.duplicate_document_count,
            "agreement_pair_count": self.agreement_pair_count,
            "exact_micro_jaccard": self.exact_micro_jaccard,
            "exact_macro_jaccard": self.exact_macro_jaccard,
            "exact_agreed_key_count": self.exact_agreed_key_count,
            "exact_union_key_count": self.exact_union_key_count,
            "source_label_agreement": {
                label: {
                    "intersection": intersection,
                    "union": union,
                    "jaccard": jaccard,
                }
                for label, intersection, union, jaccard in self.source_label_agreement
            },
            "groups": list(self.groups),
        }


@dataclass(frozen=True)
class ExactDuplicateReconciliationResult:
    """Deduplicated documents, high-confidence labels, and unresolved review labels."""

    documents: tuple[MinedDocument, ...]
    training_annotations: tuple[AnnotationProposal, ...]
    review_annotations: tuple[AnnotationProposal, ...]
    document_mappings: tuple[DocumentCanonicalMapping, ...]
    report: DuplicateReconciliationReport


def reconcile_exact_duplicates(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    *,
    labeler_id: str = "exact-duplicate-consensus:v1",
) -> ExactDuplicateReconciliationResult:
    """Collapse raw-identical documents and retain only unanimous duplicate labels.

    Single-source documents keep their imported silver proposals. For duplicate texts,
    exact semantic intersection becomes silver training data while every disagreement is
    preserved as a bronze review proposal. Near-duplicates are deliberately excluded
    because their offsets do not share a safe coordinate system.
    """

    if not labeler_id.strip():
        raise ValueError("Reconciliation labeler_id must be non-empty")
    issues = MiningQualityGate().validate(documents, annotations)
    if issues:
        raise ValueError("Cannot reconcile invalid mined data:\n" + "\n".join(issues))

    documents_by_hash: dict[str, list[MinedDocument]] = defaultdict(list)
    for document in documents:
        documents_by_hash[document.text_sha256].append(document)
    annotations_by_document: dict[str, list[AnnotationProposal]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_document[annotation.document_id].append(annotation)

    output_documents: list[MinedDocument] = []
    training_annotations: list[AnnotationProposal] = []
    review_annotations: list[AnnotationProposal] = []
    mappings: list[DocumentCanonicalMapping] = []
    group_reports: list[dict[str, Any]] = []
    pair_scores: list[float] = []
    total_pair_intersection = 0
    total_pair_union = 0
    label_pair_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    duplicate_group_count = 0
    duplicate_document_count = 0

    for text_sha256, raw_group in sorted(documents_by_hash.items()):
        group = tuple(sorted(raw_group, key=lambda item: item.document_id))
        canonical = _canonical_document(group)
        output_documents.append(canonical)
        for document in group:
            mappings.append(
                DocumentCanonicalMapping(
                    document_id=document.document_id,
                    canonical_document_id=canonical.document_id,
                    text_sha256=text_sha256,
                    group_size=len(group),
                )
            )

        if len(group) == 1:
            training_annotations.extend(annotations_by_document[group[0].document_id])
            continue

        duplicate_group_count += 1
        duplicate_document_count += len(group)
        semantic_by_document = {
            document.document_id: _semantic_proposals(annotations_by_document[document.document_id])
            for document in group
        }
        all_keys = set().union(*(set(values) for values in semantic_by_document.values()))
        unanimous_count = 0
        disagreement_count = 0
        for semantic_key in sorted(all_keys):
            votes = {
                document_id: values[semantic_key]
                for document_id, values in semantic_by_document.items()
                if semantic_key in values
            }
            source_annotations = tuple(
                annotation for values in votes.values() for annotation in values
            )
            if len(votes) == len(group):
                unanimous_count += 1
                training_annotations.append(
                    _materialize_annotation(
                        canonical,
                        semantic_key,
                        source_annotations,
                        source_document_count=len(group),
                        labeler_id=labeler_id,
                        consensus=True,
                    )
                )
            else:
                disagreement_count += 1
                review_annotations.append(
                    _materialize_annotation(
                        canonical,
                        semantic_key,
                        source_annotations,
                        source_document_count=len(group),
                        labeler_id=labeler_id,
                        consensus=False,
                    )
                )

        group_pair_scores = []
        for left, right in combinations(group, 2):
            left_keys = set(semantic_by_document[left.document_id])
            right_keys = set(semantic_by_document[right.document_id])
            intersection = len(left_keys & right_keys)
            union = len(left_keys | right_keys)
            score = _jaccard(intersection, union)
            pair_scores.append(score)
            group_pair_scores.append(score)
            total_pair_intersection += intersection
            total_pair_union += union
            for source_label in sorted({_source_label(key) for key in left_keys | right_keys}):
                left_label_keys = {key for key in left_keys if _source_label(key) == source_label}
                right_label_keys = {key for key in right_keys if _source_label(key) == source_label}
                label_pair_counts[source_label][0] += len(left_label_keys & right_label_keys)
                label_pair_counts[source_label][1] += len(left_label_keys | right_label_keys)

        group_reports.append(
            {
                "text_sha256": text_sha256,
                "canonical_document_id": canonical.document_id,
                "source_document_ids": [document.document_id for document in group],
                "external_ids": [
                    document.metadata.get("external_id", document.document_id) for document in group
                ],
                "annotation_counts": {
                    document.document_id: len(semantic_by_document[document.document_id])
                    for document in group
                },
                "unanimous_annotation_count": unanimous_count,
                "review_annotation_count": disagreement_count,
                "exact_macro_jaccard": _mean_or_none(group_pair_scores),
            }
        )

    ordered_documents = tuple(sorted(output_documents, key=lambda item: item.document_id))
    ordered_training = tuple(sorted(training_annotations, key=lambda item: item.annotation_id))
    ordered_review = tuple(sorted(review_annotations, key=lambda item: item.annotation_id))
    output_issues = MiningQualityGate().validate(
        ordered_documents, (*ordered_training, *ordered_review)
    )
    if output_issues:
        raise ValueError("Reconciliation produced invalid mined data:\n" + "\n".join(output_issues))

    label_agreement = tuple(
        (
            label,
            counts[0],
            counts[1],
            _jaccard(counts[0], counts[1]),
        )
        for label, counts in sorted(label_pair_counts.items())
    )
    report = DuplicateReconciliationReport(
        input_document_count=len(documents),
        input_annotation_count=len(annotations),
        output_document_count=len(ordered_documents),
        output_training_annotation_count=len(ordered_training),
        review_annotation_count=len(ordered_review),
        duplicate_group_count=duplicate_group_count,
        duplicate_document_count=duplicate_document_count,
        agreement_pair_count=len(pair_scores),
        exact_micro_jaccard=(
            _jaccard(total_pair_intersection, total_pair_union) if pair_scores else None
        ),
        exact_macro_jaccard=_mean_or_none(pair_scores),
        exact_agreed_key_count=total_pair_intersection,
        exact_union_key_count=total_pair_union,
        source_label_agreement=label_agreement,
        groups=tuple(group_reports),
    )
    return ExactDuplicateReconciliationResult(
        documents=ordered_documents,
        training_annotations=ordered_training,
        review_annotations=ordered_review,
        document_mappings=tuple(sorted(mappings, key=lambda item: item.document_id)),
        report=report,
    )


def _canonical_document(group: Sequence[MinedDocument]) -> MinedDocument:
    canonical = min(group, key=lambda item: item.document_id)
    if len(group) == 1:
        return canonical
    source_document_ids = [document.document_id for document in group]
    external_ids = [
        document.metadata.get("external_id", document.document_id) for document in group
    ]
    # PRIVACY: merged records inherit the most restrictive source policy.
    access_class = max(group, key=lambda item: _access_rank(item.access_class)).access_class
    redistribution = max(
        group, key=lambda item: _redistribution_rank(item.redistribution)
    ).redistribution
    parents = {document.parent_document_id for document in group}
    return replace(
        canonical,
        access_class=access_class,
        redistribution=redistribution,
        hosted_processing_allowed=all(document.hosted_processing_allowed for document in group),
        parent_document_id=parents.pop() if len(parents) == 1 else None,
        group_ids=tuple(sorted({value for document in group for value in document.group_ids})),
        metadata={
            **canonical.metadata,
            "deduplication_method": "exact_raw_text_sha256",
            "deduplicated_document_ids": _json(source_document_ids),
            "deduplicated_external_ids": _json(external_ids),
            "deduplicated_source_artifact_ids": _json(
                sorted({document.source_artifact_id for document in group})
            ),
            "deduplicated_document_count": str(len(group)),
        },
    )


def _semantic_proposals(
    annotations: Sequence[AnnotationProposal],
) -> dict[_SemanticKey, tuple[AnnotationProposal, ...]]:
    grouped: dict[_SemanticKey, list[AnnotationProposal]] = defaultdict(list)
    for annotation in annotations:
        grouped[_semantic_key(annotation)].append(annotation)
    return {
        key: tuple(sorted(values, key=lambda item: item.annotation_id))
        for key, values in grouped.items()
    }


def _semantic_key(annotation: AnnotationProposal) -> _SemanticKey:
    return (
        annotation.span,
        annotation.text,
        annotation.entity_type,
        tuple(sorted(annotation.assertions)),
        tuple(
            sorted(
                (concept.code_system, concept.code, concept.terminology_version)
                for concept in annotation.concepts
            )
        ),
        annotation.source_label or "",
        # INVARIANT: discontinuous BRAT labels sharing an envelope are not equivalent
        # unless their source segments are also identical.
        annotation.metadata.get("brat_segments", ""),
    )


def _materialize_annotation(
    document: MinedDocument,
    key: _SemanticKey,
    source_annotations: Sequence[AnnotationProposal],
    *,
    source_document_count: int,
    labeler_id: str,
    consensus: bool,
) -> AnnotationProposal:
    span, text, entity_type, assertions, concept_keys, source_label, segment_signature = key
    source_document_ids = sorted({annotation.document_id for annotation in source_annotations})
    source_annotation_ids = sorted(annotation.annotation_id for annotation in source_annotations)
    vote_count = len(source_document_ids)
    identity = _json(
        {
            "document_id": document.document_id,
            "semantic_key": key,
            "labeler_id": labeler_id,
            "consensus": consensus,
        }
    )
    concepts = []
    for code_system, code, terminology_version in concept_keys:
        confidences = [
            concept.confidence
            for annotation in source_annotations
            for concept in annotation.concepts
            if (
                concept.code_system,
                concept.code,
                concept.terminology_version,
            )
            == (code_system, code, terminology_version)
        ]
        concepts.append(
            ConceptLink(
                code_system=code_system,
                code=code,
                terminology_version=terminology_version,
                confidence=sum(confidences) / len(confidences),
            )
        )
    metadata = {
        "consensus": str(consensus).lower(),
        "vote_count": str(vote_count),
        "source_document_count": str(source_document_count),
        "vote_fraction": f"{vote_count / source_document_count:.6f}",
        "source_document_ids": _json(source_document_ids),
        "source_annotation_ids": _json(source_annotation_ids),
    }
    if segment_signature:
        metadata["brat_segments"] = segment_signature
        metadata["discontinuous"] = str(
            any(
                annotation.metadata.get("discontinuous") == "true"
                for annotation in source_annotations
            )
        ).lower()
    annotation = AnnotationProposal(
        annotation_id=("dedup-consensus:" if consensus else "dedup-review:")
        + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        document_id=document.document_id,
        span=span,
        text=text,
        entity_type=entity_type,
        assertions=assertions,
        concepts=tuple(concepts),
        confidence=sum(annotation.confidence for annotation in source_annotations)
        / len(source_annotations),
        layer=AnnotationLayer.SILVER if consensus else AnnotationLayer.BRONZE,
        label_source=("exact_duplicate_consensus" if consensus else "exact_duplicate_disagreement"),
        labeler_id=labeler_id,
        review_status=(ReviewStatus.PROPOSED if consensus else ReviewStatus.NEEDS_REVIEW),
        source_label=source_label or None,
        metadata=metadata,
    )
    annotation.validate_offsets(document)
    return annotation


def _source_label(key: _SemanticKey) -> str:
    return key[5] or "<none>"


def _access_rank(value: AccessClass) -> int:
    return {
        AccessClass.OPEN: 0,
        AccessClass.OPEN_WITH_TERMS: 1,
        AccessClass.CREDENTIALLED: 2,
        AccessClass.LOCAL_PRIVATE: 3,
        AccessClass.DUA: 4,
        AccessClass.QUARANTINE: 5,
    }[value]


def _redistribution_rank(value: RedistributionPolicy) -> int:
    return {
        RedistributionPolicy.ALLOWED: 0,
        RedistributionPolicy.ATTRIBUTION: 1,
        RedistributionPolicy.NON_COMMERCIAL: 2,
        RedistributionPolicy.PROHIBITED: 3,
        RedistributionPolicy.UNKNOWN: 4,
    }[value]


def _jaccard(intersection: int, union: int) -> float:
    return 1.0 if union == 0 else intersection / union


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
