"""Frozen mining split selection tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
)
from medical_kg_nlp.mining.splits import (
    load_split_document_ids,
    select_mined_records,
    select_mined_records_with_metadata,
)


def _document(
    document_id: str,
    *,
    metadata: dict[str, str] | None = None,
) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text="lao phổi",
        language="vi",
        note_type="biomedical_literature",
        source_artifact_id="source:fixture",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        metadata=metadata or {},
    )


def _annotation(document_id: str) -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=f"annotation:{document_id}",
        document_id=document_id,
        span=(0, 8),
        text="lao phổi",
        entity_type="FINDING",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="source_human_annotation",
        labeler_id="fixture",
        source_label="Symptom_and_Disease",
    )


def test_split_selection_preserves_order_offsets_and_annotation_identity(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"splits": {"train-1": "train", "dev-1": "development"}}),
        encoding="utf-8",
    )
    documents = (_document("train-1"), _document("dev-1"))
    annotations = (_annotation("train-1"), _annotation("dev-1"))

    ids = load_split_document_ids(manifest, "development")
    selection = select_mined_records(documents, annotations, ids)

    assert [document.document_id for document in selection.documents] == ["dev-1"]
    assert [annotation.annotation_id for annotation in selection.annotations] == [
        "annotation:dev-1"
    ]
    selection.annotations[0].validate_offsets(selection.documents[0])


def test_split_selection_rejects_stale_manifest_document_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"splits": {"missing": "development"}}),
        encoding="utf-8",
    )

    ids = load_split_document_ids(manifest, "development")
    with pytest.raises(ValueError, match="unknown documents"):
        select_mined_records((_document("known"),), (), ids)


def test_metadata_selection_keeps_only_records_with_complete_annotation_coverage() -> None:
    structured = _document("structured", metadata={"spl_fields": "[]"})
    narrative = _document("narrative")
    annotations = (_annotation("structured"), _annotation("narrative"))

    selection = select_mined_records_with_metadata(
        (structured, narrative),
        annotations,
        ("spl_fields",),
    )

    assert selection.documents == (structured,)
    assert [row.document_id for row in selection.annotations] == ["structured"]
    selection.annotations[0].validate_offsets(selection.documents[0])


def test_metadata_selection_rejects_an_empty_evaluation_population() -> None:
    with pytest.raises(ValueError, match="No mined documents"):
        select_mined_records_with_metadata(
            (_document("narrative"),),
            (),
            ("spl_fields",),
        )
