"""Regression tests for source-safe mined NER training records."""

from __future__ import annotations

from pathlib import Path

import pytest

from medical_kg_nlp.mining.model_dataset import (
    SpanDatasetConfig,
    export_span_dataset,
    iter_span_training_records,
)
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
)


def _document(text: str = "alpha beta disease gamma delta") -> MinedDocument:
    return MinedDocument(
        document_id="doc-1",
        text=text,
        language="vi",
        note_type="clinical_note",
        source_artifact_id="artifact-1",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
    )


def _annotation(
    annotation_id: str = "ann-1",
    span: tuple[int, int] = (11, 18),
) -> AnnotationProposal:
    document = _document()
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document.document_id,
        span=span,
        text=document.text[span[0] : span[1]],
        entity_type="DISEASE",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.GOLD,
        label_source="human",
        labeler_id="reviewer-v1",
    )


def test_span_records_preserve_local_and_source_offsets_across_chunks() -> None:
    document = _document()
    records = list(
        iter_span_training_records(
            (document,),
            (_annotation(),),
            {document.document_id: "train"},
            SpanDatasetConfig(max_characters=128),
        )
    )

    assert len(records) == 1
    entity = records[0]["entities"][0]
    assert records[0]["text"][entity["start"] : entity["end"]] == "disease"
    assert document.text[entity["source_start"] : entity["source_end"]] == "disease"


def test_chunk_boundary_grows_instead_of_splitting_an_entity() -> None:
    text = "a" * 120 + "target entity" + "b" * 140
    document = _document(text)
    annotation = AnnotationProposal(
        annotation_id="ann-crossing",
        document_id=document.document_id,
        span=(120, 133),
        text="target entity",
        entity_type="FINDING",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.GOLD,
        label_source="human",
        labeler_id="reviewer-v1",
    )

    records = list(
        iter_span_training_records(
            (document,),
            (annotation,),
            {document.document_id: "train"},
            SpanDatasetConfig(max_characters=128),
        )
    )

    assert records[0]["source_span"] == [0, 133]
    assert records[0]["entities"][0]["text"] == "target entity"
    assert sum(len(record["entities"]) for record in records) == 1


def test_span_records_reject_overlapping_bio_labels() -> None:
    document = _document()
    overlapping = _annotation("ann-2", (15, 23))

    with pytest.raises(ValueError, match="not BIO-compatible"):
        list(
            iter_span_training_records(
                (document,),
                (_annotation(), overlapping),
                {document.document_id: "train"},
                SpanDatasetConfig(),
            )
        )


def test_span_dataset_export_writes_pinned_manifest(tmp_path: Path) -> None:
    document = _document()
    documents_path = tmp_path / "documents.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    split_path = tmp_path / "splits.json"
    output = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    documents_path.write_text("documents\n", encoding="utf-8")
    annotations_path.write_text("annotations\n", encoding="utf-8")
    split_path.write_text("splits\n", encoding="utf-8")

    manifest = export_span_dataset(
        (document,),
        (_annotation(),),
        {document.document_id: "development"},
        SpanDatasetConfig(max_characters=128),
        output_path=output,
        manifest_path=manifest_path,
        documents_path=documents_path,
        annotations_path=annotations_path,
        split_manifest_path=split_path,
    )

    assert manifest["chunk_count"] == 1
    assert manifest["entity_count"] == 1
    assert manifest["split_chunk_counts"] == {"development": 1}
    assert manifest["output_sha256"]
    assert manifest_path.exists()
