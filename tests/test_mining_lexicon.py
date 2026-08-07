"""Mined mention inventory tests for provenance and semantic conflicts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from clingrounder.cli.main import main
from clingrounder.mining.io import write_jsonl
from clingrounder.mining.lexicon import build_mention_inventory
from clingrounder.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)


def _document(document_id: str, text: str) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text=text,
        language="vi",
        note_type="clinical_note",
        source_artifact_id="source:fixture",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
    )


def _annotation(
    document: MinedDocument,
    annotation_id: str,
    *,
    entity_type: str = "FINDING",
    source_label: str = "Symptom_and_Disease",
) -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document.document_id,
        span=(0, len(document.text)),
        text=document.text,
        entity_type=entity_type,
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="source_human_annotation",
        labeler_id="fixture-labeler",
        review_status=ReviewStatus.PROPOSED,
        source_label=source_label,
    )


def test_inventory_aggregates_normalized_surfaces_and_consensus_support() -> None:
    first = _document("first", "Lao phổi")
    second = _document("second", "lao phổi")
    first_annotation = _annotation(first, "first-ann")
    second_annotation = replace(
        _annotation(second, "second-ann"),
        label_source="exact_duplicate_consensus",
        metadata={"consensus": "true"},
    )

    result = build_mention_inventory((first, second), (first_annotation, second_annotation))

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.normalized_mention == "lao phổi"
    assert entry.occurrence_count == 2
    assert entry.document_count == 2
    assert entry.consensus_occurrence_count == 1
    assert entry.review_tier == "duplicate_consensus_supported"
    assert entry.to_dict()["concept_status"] == "unlinked"


def test_inventory_reports_same_surface_with_conflicting_semantics() -> None:
    document = _document("doc", "PCR")
    finding = _annotation(document, "finding")
    procedure = _annotation(
        document,
        "procedure",
        entity_type="PROCEDURE",
        source_label="DiagnosticProcedure",
    )

    result = build_mention_inventory((document,), (finding, procedure))

    assert len(result.entries) == 2
    assert len(result.conflicts) == 1
    assert result.conflicts[0]["conflict_types"] == [
        "entity_type_conflict",
        "source_label_conflict",
    ]


def test_lexicon_build_cli_writes_inventory_conflicts_and_report(tmp_path: Path, capsys) -> None:
    document = _document("doc", "PCR")
    annotation = _annotation(
        document,
        "annotation",
        entity_type="PROCEDURE",
        source_label="DiagnosticProcedure",
    )
    documents_path = tmp_path / "documents.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    write_jsonl(documents_path, (document.to_dict(),))
    write_jsonl(annotations_path, (annotation.to_dict(),))

    exit_code = main(
        [
            "data",
            "lexicon",
            "build",
            "--documents",
            str(documents_path),
            "--annotations",
            str(annotations_path),
            "--output",
            str(tmp_path / "inventory.jsonl"),
            "--conflicts-output",
            str(tmp_path / "conflicts.jsonl"),
            "--report-output",
            str(tmp_path / "report.json"),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    inventory = [
        json.loads(line) for line in (tmp_path / "inventory.jsonl").read_text().splitlines()
    ]
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output["entry_count"] == 1
    assert inventory[0]["normalized_mention"] == "pcr"
    assert report["inputs"]["documents_sha256"]
    assert report["outputs"]["inventory_sha256"]


def test_lexicon_build_cli_selects_only_frozen_training_records(
    tmp_path: Path,
    capsys,
) -> None:
    train_document = _document("train", "Lao phổi")
    development_document = _document("development", "PCR")
    documents_path = tmp_path / "documents.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    manifest_path = tmp_path / "manifest.json"
    write_jsonl(
        documents_path,
        (train_document.to_dict(), development_document.to_dict()),
    )
    write_jsonl(
        annotations_path,
        (
            _annotation(train_document, "train-annotation").to_dict(),
            _annotation(
                development_document,
                "development-annotation",
                entity_type="PROCEDURE",
                source_label="DiagnosticProcedure",
            ).to_dict(),
        ),
    )
    manifest_path.write_text(
        json.dumps({"splits": {"train": "train", "development": "development"}}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "data",
            "lexicon",
            "build",
            "--documents",
            str(documents_path),
            "--annotations",
            str(annotations_path),
            "--split-manifest",
            str(manifest_path),
            "--split",
            "train",
            "--output",
            str(tmp_path / "inventory.jsonl"),
            "--conflicts-output",
            str(tmp_path / "conflicts.jsonl"),
            "--report-output",
            str(tmp_path / "report.json"),
        ]
    )
    capsys.readouterr()
    inventory = [
        json.loads(line) for line in (tmp_path / "inventory.jsonl").read_text().splitlines()
    ]

    assert exit_code == 0
    assert [row["normalized_mention"] for row in inventory] == ["lao phổi"]
