"""Dataset profile tests for distributions and structural issue reporting."""

from __future__ import annotations

import json
from pathlib import Path

from clingrounder.cli.main import main
from clingrounder.mining.io import write_jsonl
from clingrounder.mining.profile import (
    build_dataset_profile,
    profile_blocking_issue_count,
)
from clingrounder.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)


def _document(document_id: str) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text="Sốt cao",
        language="vi",
        note_type="clinical_note",
        source_artifact_id="source:fixture",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        group_ids=("exact_text:fixture",),
        metadata={"newline_normalization": "none", "parser_id": "fixture"},
    )


def _annotation(document_id: str = "doc-1") -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=f"annotation-{document_id}",
        document_id=document_id,
        span=(0, 3),
        text="Sốt",
        entity_type="FINDING",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="source_human_annotation",
        labeler_id="fixture-labeler",
        review_status=ReviewStatus.PROPOSED,
        source_label="Symptom_and_Disease",
        metadata={"discontinuous": "false"},
    )


def test_dataset_profile_reports_duplicates_labels_and_valid_offsets() -> None:
    documents = (_document("doc-1"), _document("doc-2"))

    profile = build_dataset_profile(documents, (_annotation(),))

    assert profile["documents"]["exact_duplicate_group_count"] == 1
    assert profile["documents"]["exact_duplicate_document_count"] == 2
    assert profile["annotations"]["source_labels"] == {"Symptom_and_Disease": 1}
    assert profile["annotations"]["entity_types"] == {"FINDING": 1}
    assert profile["validation"]["offset_mismatch_count"] == 0
    assert profile_blocking_issue_count(profile) == 0


def test_dataset_inspect_cli_writes_machine_readable_profile(tmp_path: Path, capsys) -> None:
    documents_path = tmp_path / "documents.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    output_path = tmp_path / "profile.json"
    write_jsonl(documents_path, (_document("doc-1").to_dict(),))
    write_jsonl(annotations_path, (_annotation().to_dict(),))

    exit_code = main(
        [
            "data",
            "dataset",
            "inspect",
            "--documents",
            str(documents_path),
            "--annotations",
            str(annotations_path),
            "--output",
            str(output_path),
            "--strict",
        ]
    )
    command_output = json.loads(capsys.readouterr().out)
    profile = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert command_output["blocking_issue_count"] == 0
    assert profile["schema_version"] == "medical-mining-profile.v1"
