"""All-authorized Phase 1 token supervision must retain LF offsets and provenance."""

from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.final_supervision import (
    Phase1FinalSupervisionCorpus,
)
from medical_kg_nlp.benchmarks.phase1.final_token_dataset import (
    Phase1FinalTokenDatasetConfig,
    build_phase1_final_token_dataset,
)
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus


def test_final_token_dataset_exports_all_authorized_rows_with_exact_offsets(tmp_path: Path) -> None:
    text, rows = _five_type_rows()
    corpus = Phase1FinalSupervisionCorpus(
        reviewed=Phase1ReviewedCorpus(
            source_texts={"authorized:1": text},
            gold_rows={"authorized:1": tuple(rows)},
            split_by_document={"authorized:1": "train"},
        ),
        source_by_document={"authorized:1": "authorized_ground_truth"},
        manifest={"fingerprint_sha256": "a" * 64},
    )
    output = tmp_path / "final-token"

    report = build_phase1_final_token_dataset(
        corpus,
        output,
        config=Phase1FinalTokenDatasetConfig(max_characters=128),
    )
    repeated = build_phase1_final_token_dataset(
        corpus,
        output,
        config=Phase1FinalTokenDatasetConfig(max_characters=128),
    )

    assert repeated == report
    assert report["dataset"]["document_count"] == 1
    assert report["dataset"]["annotation_count"] == 5
    assert report["build_contract"]["round2_included"] is False
    assert report["build_contract"]["friend31_included"] is False
    records = [json.loads(line) for line in (output / "spans.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["split"] == "train"
    for entity in records[0]["entities"]:
        assert records[0]["text"][entity["start"] : entity["end"]] == entity["text"]


def _five_type_rows() -> tuple[str, list[dict[str, object]]]:
    mentions = (
        ("triệu chứng", "TRIỆU_CHỨNG"),
        ("xét nghiệm", "TÊN_XÉT_NGHIỆM"),
        ("kết quả", "KẾT_QUẢ_XÉT_NGHIỆM"),
        ("chẩn đoán", "CHẨN_ĐOÁN"),
        ("thuốc", "THUỐC"),
    )
    text = " | ".join(mention for mention, _ in mentions)
    rows: list[dict[str, object]] = []
    cursor = 0
    for mention, entity_type in mentions:
        start = text.index(mention, cursor)
        end = start + len(mention)
        rows.append(
            {
                "text": mention,
                "position": [start, end],
                "type": entity_type,
                "assertions": [],
                "candidates": [],
            }
        )
        cursor = end
    return text, rows
