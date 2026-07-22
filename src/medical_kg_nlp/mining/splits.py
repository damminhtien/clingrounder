"""Select immutable mined records from a frozen snapshot split manifest."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.mining.records import AnnotationProposal, MinedDocument

__all__ = [
    "MinedRecordSelection",
    "load_split_document_ids",
    "select_mined_records",
    "select_mined_records_with_metadata",
]


@dataclass(frozen=True)
class MinedRecordSelection:
    """Documents and annotations selected without changing source record identity."""

    documents: tuple[MinedDocument, ...]
    annotations: tuple[AnnotationProposal, ...]


def load_split_document_ids(
    manifest_path: str | Path,
    split: str,
) -> frozenset[str]:
    """Load one named split from an immutable snapshot manifest."""

    if not split.strip():
        raise ValueError("Snapshot split name must be non-empty")
    raw: Any = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Snapshot manifest must be an object")
    raw_splits = raw.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise ValueError("Snapshot manifest requires a splits object")
    invalid_rows = [
        (document_id, value)
        for document_id, value in raw_splits.items()
        if not isinstance(document_id, str) or not isinstance(value, str)
    ]
    if invalid_rows:
        raise ValueError("Snapshot split entries must map string IDs to string names")
    document_ids = frozenset(
        document_id
        for document_id, value in raw_splits.items()
        if value == split
    )
    if not document_ids:
        available = sorted({str(value) for value in raw_splits.values()})
        raise ValueError(
            f"Snapshot split {split!r} is empty or unknown; available={available}"
        )
    return document_ids


def select_mined_records(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    document_ids: frozenset[str],
) -> MinedRecordSelection:
    """Select records while rejecting stale manifests and unknown annotation references."""

    documents_by_id = {document.document_id: document for document in documents}
    missing_ids = sorted(document_ids - documents_by_id.keys())
    if missing_ids:
        raise ValueError(f"Snapshot manifest references unknown documents: {missing_ids[:5]}")
    unknown_annotation_ids = sorted(
        {
            annotation.document_id
            for annotation in annotations
            if annotation.document_id not in documents_by_id
        }
    )
    if unknown_annotation_ids:
        raise ValueError(
            "Annotations reference unknown documents: "
            f"{unknown_annotation_ids[:5]}"
        )

    # INVARIANT: selection preserves source order, annotation IDs, and raw offsets.
    selected_documents = tuple(
        document for document in documents if document.document_id in document_ids
    )
    selected_annotations = tuple(
        annotation
        for annotation in annotations
        if annotation.document_id in document_ids
    )
    return MinedRecordSelection(selected_documents, selected_annotations)


def select_mined_records_with_metadata(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    required_keys: Sequence[str],
) -> MinedRecordSelection:
    """Keep documents carrying all required metadata and their annotations.

    Some source parsers emit several representations with different annotation coverage. For
    example, a structured product record may be exhaustively labeled while its sibling narrative
    record is not. Evaluating both as if they had the same gold coverage would turn unlabeled
    mentions into false positives.
    """

    normalized_keys = tuple(dict.fromkeys(key.strip() for key in required_keys))
    if not normalized_keys or any(not key for key in normalized_keys):
        raise ValueError("Required document metadata keys must be non-empty")
    selected_ids = frozenset(
        document.document_id
        for document in documents
        if all(key in document.metadata for key in normalized_keys)
    )
    if not selected_ids:
        raise ValueError(
            "No mined documents contain all required metadata keys: "
            f"{list(normalized_keys)}"
        )
    # INVARIANT: delegate record selection so annotation identity and raw spans remain unchanged.
    return select_mined_records(documents, annotations, selected_ids)
