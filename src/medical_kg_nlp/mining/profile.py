"""Task-neutral corpus profiling for mined documents and annotations."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from medical_kg_nlp.mining.records import AnnotationProposal, MinedDocument

__all__ = ["build_dataset_profile", "profile_blocking_issue_count"]


def build_dataset_profile(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal] = (),
) -> dict[str, Any]:
    """Build a deterministic source profile and audit raw annotation offsets.

    The report intentionally describes source labels separately from internal entity
    types. This keeps imported corpora comparable without pretending their annotation
    conventions are equivalent.
    """

    document_id_counts = Counter(document.document_id for document in documents)
    annotation_id_counts = Counter(annotation.annotation_id for annotation in annotations)
    documents_by_id = {document.document_id: document for document in documents}
    annotations_by_document = Counter(annotation.document_id for annotation in annotations)

    unknown_document_references: list[str] = []
    offset_mismatches: list[str] = []
    for annotation in annotations:
        document = documents_by_id.get(annotation.document_id)
        if document is None:
            unknown_document_references.append(annotation.annotation_id)
            continue
        try:
            # INVARIANT: every imported annotation is audited against immutable source text.
            annotation.validate_offsets(document)
        except ValueError:
            offset_mismatches.append(annotation.annotation_id)

    text_counts = Counter(document.text_sha256 for document in documents)
    duplicate_group_sizes = sorted(count for count in text_counts.values() if count > 1)
    assertion_counts = Counter(
        assertion for annotation in annotations for assertion in annotation.assertions
    )
    concept_counts = Counter(
        concept.code_system for annotation in annotations for concept in annotation.concepts
    )

    return {
        "schema_version": "medical-mining-profile.v1",
        "documents": {
            "count": len(documents),
            "unique_text_count": len(text_counts),
            "exact_duplicate_group_count": len(duplicate_group_sizes),
            "exact_duplicate_document_count": sum(duplicate_group_sizes),
            "source_artifact_count": len({document.source_artifact_id for document in documents}),
            "parent_document_count": sum(
                document.parent_document_id is not None for document in documents
            ),
            "text_length": _numeric_summary(len(document.text) for document in documents),
            "annotation_count": _numeric_summary(
                annotations_by_document.get(document.document_id, 0) for document in documents
            ),
            "languages": _distribution(document.language for document in documents),
            "note_types": _distribution(document.note_type for document in documents),
            "access_classes": _distribution(document.access_class.value for document in documents),
            "redistribution": _distribution(
                document.redistribution.value for document in documents
            ),
            "parser_ids": _distribution(
                document.metadata.get("parser_id", "<unknown>") for document in documents
            ),
            "newline_modes": _distribution(
                document.metadata.get("newline_normalization", "<none>") for document in documents
            ),
        },
        "annotations": {
            "count": len(annotations),
            "document_coverage_count": len(
                {annotation.document_id for annotation in annotations} & set(documents_by_id)
            ),
            "entity_types": _distribution(annotation.entity_type for annotation in annotations),
            "source_labels": _distribution(
                annotation.source_label or "<none>" for annotation in annotations
            ),
            "label_sources": _distribution(annotation.label_source for annotation in annotations),
            "layers": _distribution(annotation.layer.value for annotation in annotations),
            "review_statuses": _distribution(
                annotation.review_status.value for annotation in annotations
            ),
            "assertions": dict(sorted(assertion_counts.items())),
            "concept_systems": dict(sorted(concept_counts.items())),
            "concept_link_count": sum(concept_counts.values()),
            "discontinuous_count": sum(
                annotation.metadata.get("discontinuous") == "true" for annotation in annotations
            ),
            "span_length": _numeric_summary(
                annotation.span[1] - annotation.span[0] for annotation in annotations
            ),
        },
        "validation": {
            "duplicate_document_id_count": sum(
                count - 1 for count in document_id_counts.values() if count > 1
            ),
            "duplicate_annotation_id_count": sum(
                count - 1 for count in annotation_id_counts.values() if count > 1
            ),
            "unknown_document_reference_count": len(unknown_document_references),
            "offset_mismatch_count": len(offset_mismatches),
            "issue_samples": {
                "unknown_document_references": sorted(unknown_document_references)[:20],
                "offset_mismatches": sorted(offset_mismatches)[:20],
            },
        },
    }


def profile_blocking_issue_count(profile: dict[str, Any]) -> int:
    """Return the number of structural issues exposed by a profile report."""

    validation = profile.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Mining profile has no validation mapping")
    fields = (
        "duplicate_document_id_count",
        "duplicate_annotation_id_count",
        "unknown_document_reference_count",
        "offset_mismatch_count",
    )
    return sum(int(validation.get(field, 0)) for field in fields)


def _distribution(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _numeric_summary(values: Iterable[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": round(statistics.fmean(ordered), 3),
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }
