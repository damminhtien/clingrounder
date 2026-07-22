"""Bounded-memory annotation materialization and CLI batching contracts."""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest

from medical_kg_nlp.cli.commands import data as data_commands
from medical_kg_nlp.mining.annotation_manifest import (
    materialize_annotation_manifest,
)
from medical_kg_nlp.mining.io import load_annotations, write_jsonl
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
)


class _BatchRecordingLabeler:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def propose(
        self,
        documents: Sequence[MinedDocument],
    ) -> Iterable[AnnotationProposal]:
        self.batch_sizes.append(len(documents))
        for document in reversed(documents):
            yield _proposal(document, annotation_id=f"annotation:{document.document_id}")


def test_annotation_manifest_external_sorts_and_deduplicates(tmp_path: Path) -> None:
    first = _document("b")
    second = _document("a")
    duplicate = _proposal(first, annotation_id="annotation:b")
    output = tmp_path / "annotations.jsonl"

    result = materialize_annotation_manifest(
        output,
        (
            duplicate,
            _proposal(second, annotation_id="annotation:a"),
            duplicate,
        ),
    )

    assert result.annotation_count == 2
    assert result.duplicate_count == 1
    assert [value.annotation_id for value in load_annotations(output)] == [
        "annotation:a",
        "annotation:b",
    ]


def test_annotation_manifest_rejects_conflicting_duplicate_id(tmp_path: Path) -> None:
    document = _document("a", text="drug dose")
    first = _proposal(document, annotation_id="same", span=(0, 4))
    second = _proposal(document, annotation_id="same", span=(5, 9))

    with pytest.raises(ValueError, match="Conflicting annotation ID 'same'"):
        materialize_annotation_manifest(tmp_path / "annotations.jsonl", (first, second))


def test_label_cli_streams_fixed_size_batches_and_keeps_global_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    documents = tuple(_document(value) for value in ("e", "d", "c", "b", "a"))
    documents_path = tmp_path / "documents.jsonl"
    output = tmp_path / "annotations.jsonl"
    write_jsonl(documents_path, (document.to_dict() for document in documents))
    labeler = _BatchRecordingLabeler()
    monkeypatch.setattr(data_commands, "_load_labeler", lambda *_: labeler)

    assert data_commands.propose_labels(
        Namespace(
            documents=str(documents_path),
            adapter="fixture:factory",
            adapter_config=None,
            hosted=False,
            output=str(output),
            batch_size=2,
        )
    ) == 0
    report = json.loads(capsys.readouterr().out)

    assert labeler.batch_sizes == [2, 2, 1]
    assert report["proposal_count"] == 5
    assert report["duplicate_count"] == 0
    assert [value.annotation_id for value in load_annotations(output)] == [
        "annotation:a",
        "annotation:b",
        "annotation:c",
        "annotation:d",
        "annotation:e",
    ]


def _document(document_id: str, *, text: str = "drug") -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text=text,
        language="en",
        note_type="structured_medication_record",
        source_artifact_id="source",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
    )


def _proposal(
    document: MinedDocument,
    *,
    annotation_id: str,
    span: tuple[int, int] = (0, 4),
) -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document.document_id,
        span=span,
        text=document.text[span[0] : span[1]],
        entity_type="DRUG",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="fixture",
        labeler_id="fixture:v1",
    )
