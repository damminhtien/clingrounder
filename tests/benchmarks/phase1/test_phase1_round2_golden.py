from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.round2_golden import (
    build_phase1_round2_golden,
    write_phase1_round2_golden,
)
from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_round2_golden_separates_consensus_from_review_and_expands_medication(
    tmp_path: Path,
) -> None:
    text = (
        "Danh sách thuốc trước nhập viện chính xác và đầy đủ.\r\n"
        "1. aspirin 81 mg po daily điều trị đau ngực và ho"
    )
    drug_start = text.index("aspirin")
    full_drug = "aspirin 81 mg po daily"
    symptom_start = text.index("đau ngực")
    cough_start = text.index("ho")
    dictionary = _dictionary()
    short_drug = _row(text, "aspirin", "THUỐC", drug_start, ["243670"])
    symptom = _row(text, "đau ngực", "TRIỆU_CHỨNG", symptom_start)
    sources = {
        "baseline": {"1": [short_drug, symptom]},
        "friend31": {"1": [short_drug, symptom]},
        "qwen": {
            "1": [
                _row(text, full_drug, "THUỐC", drug_start),
                symptom,
                _row(text, "ho", "TRIỆU_CHỨNG", cough_start),
            ]
        },
    }

    report = build_phase1_round2_golden(
        {"1": text},
        sources,
        dictionary,
        minimum_sources=2,
    )

    strict = report["gold_strict"]["1"]
    assert [(row["text"], row["type"]) for row in strict] == [
        (full_drug, "THUỐC"),
        ("đau ngực", "TRIỆU_CHỨNG"),
    ]
    assert strict[0]["candidates"] == ["243670"]
    assert strict[0]["assertions"] == ["isHistorical"]
    assert "candidates" not in strict[1]
    assert strict[1]["assertions"] == []
    assert len(report["gold_review"]["1"]) == 4
    assert {
        (row["text"], row["reason"]) for row in report["review_queue"]
    } == {
        ("aspirin", "superseded_by_btc_medication_full_span"),
        ("ho", "source_only"),
    }
    assert report["summary"]["review_group_count"] == 2

    documents = [ClinicalDocument(document_id="1", text=text)]
    manifest = write_phase1_round2_golden(
        report,
        tmp_path / "golden",
        documents=documents,
        dictionary=dictionary,
        provenance={"sources": ["baseline", "friend31", "qwen"]},
    )

    assert manifest["official_gold"] is False
    assert manifest["validation"]["strict_issue_count"] == 0
    assert (tmp_path / "golden" / "gold_strict.zip").exists()
    assert (tmp_path / "golden" / "review_queue.jsonl").exists()
    assert (tmp_path / "golden" / "review_groups.jsonl").exists()
    written = json.loads(
        (tmp_path / "golden" / "gold_strict" / "1.json").read_text(encoding="utf-8")
    )
    assert written == strict


def test_round2_golden_sends_type_conflicts_to_review() -> None:
    text = "bại não"
    symptom = _row(text, text, "TRIỆU_CHỨNG", 0)
    diagnosis = _row(text, text, "CHẨN_ĐOÁN", 0)

    report = build_phase1_round2_golden(
        {"1": text},
        {
            "baseline": {"1": [symptom]},
            "friend31": {"1": [symptom]},
            "qwen": {"1": [diagnosis]},
        },
        _dictionary(),
    )

    assert report["gold_strict"]["1"] == []
    assert len(report["gold_review"]["1"]) == 2
    assert {row["reason"] for row in report["review_queue"]} == {"type_conflict"}


def test_round2_golden_cli_accepts_independent_sources() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "round2",
            "golden",
            "--documents",
            "documents.jsonl",
            "--source-archive-sha256",
            "a" * 64,
            "--source",
            "baseline=baseline.zip",
            "--source",
            "qwen=qwen",
        ]
    )

    assert args.handler == "benchmark_phase1_round2_golden"
    assert args.source == ["baseline=baseline.zip", "qwen=qwen"]
    assert args.minimum_sources == 2


def _dictionary() -> DictionaryStore:
    return DictionaryStore(
        [
            ConceptEntry(
                concept_id="RXNORM:243670",
                code="243670",
                code_system=CodeSystem.RXNORM,
                canonical_name="aspirin 81 MG Oral Tablet",
                semantic_type=EntityType.DRUG,
                source="test",
                rxnorm_tty="SCD",
            )
        ]
    )


def _row(
    source_text: str,
    text: str,
    entity_type: str,
    start: int,
    candidates: list[str] | None = None,
) -> dict[str, object]:
    assert source_text[start : start + len(text)] == text
    row: dict[str, object] = {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "position": [start, start + len(text)],
    }
    if entity_type in {"THUỐC", "CHẨN_ĐOÁN"}:
        row["candidates"] = list(candidates or [])
    return row
