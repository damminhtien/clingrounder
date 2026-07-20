"""Disk-backed document manifest regression tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from medical_kg_nlp.mining.document_manifest import materialize_document_manifest
from medical_kg_nlp.mining.io import iter_documents
from medical_kg_nlp.mining.records import (
    AccessClass,
    MinedDocument,
    RedistributionPolicy,
)


def test_manifest_sorts_documents_and_merges_duplicate_origins(tmp_path: Path) -> None:
    second_origin = _document(
        document_id="dailymed:doc-b",
        source_artifact_id="dailymed:part-b",
        archive_member="labels/b.xml",
    )
    first_origin = replace(
        second_origin,
        source_artifact_id="dailymed:part-a",
        metadata={
            **second_origin.metadata,
            "archive_member": "labels/a.xml",
            "dailymed_source_version": "release-a",
        },
    )
    earlier = _document(
        document_id="dailymed:doc-a",
        source_artifact_id="dailymed:part-a",
        archive_member="labels/earlier.xml",
    )

    result = materialize_document_manifest(
        tmp_path / "documents.jsonl",
        (second_origin, earlier, first_origin),
    )
    documents = tuple(iter_documents(result.path))

    assert result.document_count == 2
    assert result.duplicate_count == 1
    assert [document.document_id for document in documents] == [
        "dailymed:doc-a",
        "dailymed:doc-b",
    ]
    merged = documents[1]
    assert merged.source_artifact_id == "dailymed:part-a"
    assert json.loads(merged.metadata["source_artifact_ids"]) == [
        "dailymed:part-a",
        "dailymed:part-b",
    ]
    assert json.loads(merged.metadata["source_archive_members"]) == [
        "dailymed:part-a:labels/a.xml",
        "dailymed:part-b:labels/b.xml",
    ]
    assert json.loads(merged.metadata["source_versions"]) == [
        "release-a",
        "release-b",
    ]


def test_manifest_rejects_same_id_with_different_content(tmp_path: Path) -> None:
    document = _document(
        document_id="dailymed:collision",
        source_artifact_id="dailymed:part-a",
        archive_member="labels/a.xml",
    )

    with pytest.raises(ValueError, match="Conflicting document ID"):
        materialize_document_manifest(
            tmp_path / "documents.jsonl",
            (document, replace(document, text="Different label text")),
        )


def _document(
    *,
    document_id: str,
    source_artifact_id: str,
    archive_member: str,
) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text="Stable SPL label text",
        language="en",
        note_type="structured_product_label",
        source_artifact_id=source_artifact_id,
        access_class=AccessClass.OPEN_WITH_TERMS,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        group_ids=("drug_label:set-42",),
        metadata={
            "archive_member": archive_member,
            "dailymed_source_version": "release-b",
            "external_id": "set-42",
            "parser_id": "spl_xml",
            "parser_revision": "3",
            "source_unit_sha256": "a" * 64,
        },
    )
