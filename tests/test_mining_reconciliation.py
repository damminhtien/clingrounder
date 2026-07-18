"""Exact duplicate reconciliation tests for consensus and review preservation."""

from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.cli.main import main
from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.mining.reconciliation import reconcile_exact_duplicates
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)


def _document(document_id: str, external_id: str) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text="Sốt và ho.",
        language="vi",
        note_type="clinical_note",
        source_artifact_id="source:fixture",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        group_ids=("exact_text:fixture",),
        metadata={"external_id": external_id},
    )


def _annotation(
    document_id: str,
    annotation_id: str,
    span: tuple[int, int],
    text: str,
    *,
    segments: str = "",
) -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document_id,
        span=span,
        text=text,
        entity_type="FINDING",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="source_human_annotation",
        labeler_id="source-labeler",
        review_status=ReviewStatus.PROPOSED,
        source_label="Symptom_and_Disease",
        metadata={"brat_segments": segments} if segments else {},
    )


def _fixture() -> tuple[tuple[MinedDocument, ...], tuple[AnnotationProposal, ...]]:
    left = _document("left", "Annotator_A/dup_1")
    right = _document("right", "Annotator_B/dup_1")
    annotations = (
        _annotation("left", "left-fever", (0, 3), "Sốt"),
        _annotation("right", "right-fever", (0, 3), "Sốt"),
        _annotation("left", "left-cough", (7, 9), "ho"),
    )
    return (left, right), annotations


def test_reconciliation_keeps_intersection_and_routes_union_difference_to_review() -> None:
    documents, annotations = _fixture()

    result = reconcile_exact_duplicates(documents, annotations)

    assert len(result.documents) == 1
    assert len(result.training_annotations) == 1
    assert len(result.review_annotations) == 1
    assert result.training_annotations[0].text == "Sốt"
    assert result.training_annotations[0].layer is AnnotationLayer.SILVER
    assert result.review_annotations[0].text == "ho"
    assert result.review_annotations[0].review_status is ReviewStatus.NEEDS_REVIEW
    assert result.report.exact_micro_jaccard == 0.5
    assert result.report.exact_macro_jaccard == 0.5
    for annotation in (*result.training_annotations, *result.review_annotations):
        annotation.validate_offsets(result.documents[0])


def test_reconciliation_treats_discontinuous_segment_geometry_as_semantic() -> None:
    documents, _ = _fixture()
    annotations = (
        _annotation("left", "left", (0, 9), "Sốt và ho", segments="[[0,3],[7,9]]"),
        _annotation("right", "right", (0, 9), "Sốt và ho", segments="[[0,9]]"),
    )

    result = reconcile_exact_duplicates(documents, annotations)

    assert result.training_annotations == ()
    assert len(result.review_annotations) == 2
    assert result.report.exact_micro_jaccard == 0.0


def test_reconcile_duplicates_cli_writes_all_audit_artifacts(tmp_path: Path, capsys) -> None:
    documents, annotations = _fixture()
    documents_path = tmp_path / "documents.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    write_jsonl(documents_path, (document.to_dict() for document in documents))
    write_jsonl(annotations_path, (annotation.to_dict() for annotation in annotations))

    exit_code = main(
        [
            "data",
            "dataset",
            "reconcile-duplicates",
            "--documents",
            str(documents_path),
            "--annotations",
            str(annotations_path),
            "--documents-output",
            str(tmp_path / "deduplicated-documents.jsonl"),
            "--annotations-output",
            str(tmp_path / "training-annotations.jsonl"),
            "--review-output",
            str(tmp_path / "review-annotations.jsonl"),
            "--mapping-output",
            str(tmp_path / "document-map.jsonl"),
            "--report-output",
            str(tmp_path / "report.json"),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output["document_count"] == 1
    assert report["schema_version"] == "medical-duplicate-reconciliation.v1"
    assert len((tmp_path / "document-map.jsonl").read_text().splitlines()) == 2
