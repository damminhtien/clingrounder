"""Offline tests for VietMed-NER Parquet parsing and BIO import."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from clingrounder.mining.labelers.vietmed_ner import (
    VietMedNerSourceLabelerAdapter,
    decode_vietmed_bio_spans,
)
from clingrounder.mining.parsers.vietmed_ner import (
    VietMedNerParquetParser,
    align_vietmed_tokens,
)
from clingrounder.mining.records import (
    AccessClass,
    RedistributionPolicy,
    SourceArtifact,
)
from clingrounder.mining.storage import LocalArtifactStore


def test_vietmed_parquet_parser_projects_text_columns_without_audio(
    tmp_path: Path,
) -> None:
    payload = _parquet_fixture()
    artifact, store = _artifact(tmp_path, payload)

    documents = list(VietMedNerParquetParser().parse(artifact, store=store))

    assert len(documents) == 1
    document = documents[0]
    assert document.text == "béo phì điều trị thuốc aspirin"
    assert document.metadata["source_split"] == "train"
    assert json.loads(document.metadata["token_offsets"]) == [
        [0, 3],
        [4, 7],
        [8, 12],
        [13, 16],
        [17, 22],
        [23, 30],
    ]


def test_vietmed_source_labeler_preserves_source_taxonomy(tmp_path: Path) -> None:
    artifact, store = _artifact(tmp_path, _parquet_fixture())
    document = next(iter(VietMedNerParquetParser().parse(artifact, store=store)))
    labeler = VietMedNerSourceLabelerAdapter(
        label_map={
            "DISEASESYMTOM": "DISEASESYMTOM",
            "TREATMENT": "TREATMENT",
            "DRUGCHEMICAL": "DRUGCHEMICAL",
        },
        labeler_id="vietmed-test@pinned",
    )

    proposals = list(labeler.propose((document,)))

    assert [
        (item.text, item.entity_type, item.source_label)
        for item in proposals
    ] == [
        ("béo phì", "DISEASESYMTOM", "DISEASESYMTOM"),
        ("điều trị", "TREATMENT", "TREATMENT"),
        ("aspirin", "DRUGCHEMICAL", "DRUGCHEMICAL"),
    ]


def test_vietmed_alignment_and_decoder_handle_repeated_words_and_orphan_i() -> None:
    text = "đau rồi đau"
    offsets = align_vietmed_tokens(text, ("đau", "rồi", "đau"))

    spans = decode_vietmed_bio_spans(
        offsets,
        ("I-DISEASESYMTOM", "0", "B-DISEASESYMTOM"),
    )

    assert [(item.span, item.source_label) for item in spans] == [
        ((0, 3), "DISEASESYMTOM"),
        ((8, 11), "DISEASESYMTOM"),
    ]


def _parquet_fixture() -> bytes:
    table = pa.table(
        {
            "words": [
                ["béo", "phì", "điều", "trị", "thuốc", "aspirin"],
            ],
            "labels": [
                [
                    "B-DISEASESYMTOM",
                    "I-DISEASESYMTOM",
                    "B-TREATMENT",
                    "I-TREATMENT",
                    "0",
                    "B-DRUGCHEMICAL",
                ]
            ],
            "text": ["béo phì điều trị thuốc aspirin"],
            "duration": [4.0],
            # This deliberately large-looking column is not requested by the parser.
            "audio": [b"unused-audio-bytes"],
        }
    )
    stream = io.BytesIO()
    pq.write_table(table, stream)
    return stream.getvalue()


def _artifact(
    root: Path,
    payload: bytes,
) -> tuple[SourceArtifact, LocalArtifactStore]:
    store = LocalArtifactStore(root / "store")
    stored = store.put_stream(io.BytesIO(payload), metadata={})
    artifact = SourceArtifact(
        artifact_id="vietmed_ner:fixture",
        source_id="vietmed_ner",
        source_version="hf-pinned",
        source_uri="memory://vietmed",
        object=stored,
        media_type="application/vnd.apache.parquet",
        license_id="research-use-confirmed",
        access_class=AccessClass.OPEN_WITH_TERMS,
        redistribution=RedistributionPolicy.PROHIBITED,
        hosted_processing_allowed=True,
        retrieved_at="2026-07-27T00:00:00+00:00",
        metadata={"split": "train"},
    )
    return artifact, store
