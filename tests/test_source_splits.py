"""Tests for preserving source-defined dataset partitions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.mining.records import (
    AccessClass,
    MinedDocument,
    RedistributionPolicy,
)
from medical_kg_nlp.mining.source_splits import freeze_source_splits


def test_freeze_source_splits_preserves_declared_assignments(tmp_path: Path) -> None:
    documents = (
        _document("train-doc", "train"),
        _document("validation-doc", "validation"),
        _document("test-doc", "test"),
    )
    documents_path = tmp_path / "documents.jsonl"
    write_jsonl(documents_path, (document.to_dict() for document in documents))
    output_path = tmp_path / "splits.json"

    report = freeze_source_splits(
        documents,
        metadata_key="source_split",
        split_map={
            "train": "train",
            "validation": "development",
            "test": "test",
        },
        documents_path=documents_path,
        output_path=output_path,
    )

    assert report["split_counts"] == {
        "development": 1,
        "test": 1,
        "train": 1,
    }
    assert report["splits"] == {
        "test-doc": "test",
        "train-doc": "train",
        "validation-doc": "development",
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


def test_freeze_source_splits_rejects_unmapped_source_partition(
    tmp_path: Path,
) -> None:
    documents = (_document("hidden-doc", "hidden"),)
    documents_path = tmp_path / "documents.jsonl"
    write_jsonl(documents_path, (document.to_dict() for document in documents))

    with pytest.raises(ValueError, match="Unmapped source split"):
        freeze_source_splits(
            documents,
            metadata_key="source_split",
            split_map={"train": "train"},
            documents_path=documents_path,
            output_path=tmp_path / "splits.json",
        )


def _document(document_id: str, split: str) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text="bệnh nhân đau đầu",
        language="vi",
        note_type="spoken_medical_transcript",
        source_artifact_id=f"artifact:{split}",
        access_class=AccessClass.OPEN_WITH_TERMS,
        redistribution=RedistributionPolicy.PROHIBITED,
        hosted_processing_allowed=True,
        metadata={"source_split": split},
    )
