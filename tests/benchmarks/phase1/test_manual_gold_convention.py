from __future__ import annotations

import json
from pathlib import Path

import pytest

from clingrounder.benchmarks.phase1.manual_gold_convention import (
    audit_manual_gold_convention,
    write_manual_gold_convention_report,
)


def test_btc_medication_list_convention_passes(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    source = Path("tests/fixtures/phase1/btc_medication_list_crlf.txt")
    expected = Path("tests/fixtures/phase1/btc_medication_list_expected.json")
    (input_dir / "1.txt").write_bytes(source.read_bytes())
    (gold_dir / "1.json").write_bytes(expected.read_bytes())

    report = audit_manual_gold_convention(input_dir, gold_dir, expected_count=1)

    assert report["blocking_count"] == 0
    assert report["review_count"] == 0
    assert report["entity_count"] == 19


def test_audit_reports_boundary_assertion_and_cardinality_conflicts(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    text = "Danh sách thuốc trước nhập viện.\n1. aspirin 81 mg po daily điều trị đau"
    (input_dir / "1.txt").write_text(text, encoding="utf-8")
    start = text.index("aspirin")
    symptom_start = text.index("đau")
    rows = [
        {
            "text": "aspirin",
            "type": "THUỐC",
            "assertions": [],
            "candidates": ["1191", "243670"],
            "position": [start, start + len("aspirin")],
        },
        {
            "text": "đau",
            "type": "TRIỆU_CHỨNG",
            "assertions": ["isHistorical"],
            "position": [symptom_start, symptom_start + len("đau")],
        },
    ]
    (gold_dir / "1.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    report = audit_manual_gold_convention(input_dir, gold_dir, expected_count=1)

    assert report["by_kind"] == {
        "indication_assertion": 1,
        "medication_boundary_under": 1,
        "medication_list_boundary": 1,
        "medication_list_history": 1,
        "multi_candidate_review": 1,
    }
    assert report["blocking_count"] == 3
    assert report["review_count"] == 2

    output_dir = tmp_path / "report"
    write_manual_gold_convention_report(report, output_dir)
    assert (output_dir / "audit.json").exists()
    assert "`medication_list_boundary`: 1" in (output_dir / "report.md").read_text(encoding="utf-8")


def test_audit_accepts_reviewed_concept_decision_without_document_specific_rule(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    text = "Dùng ciproflagyl."
    (input_dir / "1.txt").write_text(text, encoding="utf-8")
    start = text.index("ciproflagyl")
    (gold_dir / "1.json").write_text(
        json.dumps(
            [
                {
                    "text": "ciproflagyl",
                    "type": "THUỐC",
                    "assertions": [],
                    "candidates": ["2551", "6922"],
                    "position": [start, start + len("ciproflagyl")],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    decisions = gold_dir / "convention_decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "kind": "multi_candidate_review",
                "entity_type": "THUỐC",
                "normalized_mention": "ciproflagyl",
                "decision": "allow",
                "reason": "The raw token fuses two drug names.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_manual_gold_convention(input_dir, gold_dir, expected_count=1)

    assert report["review_count"] == 0
    assert report["resolved_count"] == 1
    assert report["resolutions"][0]["decision"]["reason"]


def test_audit_rejects_document_specific_convention_decision(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    (input_dir / "1.txt").write_text("aspirin", encoding="utf-8")
    (gold_dir / "1.json").write_text("[]", encoding="utf-8")
    (gold_dir / "convention_decisions.jsonl").write_text(
        json.dumps(
            {
                "kind": "multi_candidate_review",
                "entity_type": "THUỐC",
                "normalized_mention": "aspirin",
                "decision": "allow",
                "reason": "invalid test row",
                "document_id": "1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="document-specific"):
        audit_manual_gold_convention(input_dir, gold_dir, expected_count=1)


def test_audit_blocks_nested_overlapping_entities(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    text = "Viêm gan virus C và B"
    (input_dir / "1.txt").write_text(text, encoding="utf-8")
    (gold_dir / "1.json").write_text(
        json.dumps(
            [
                {
                    "text": "Viêm gan virus C",
                    "type": "CHẨN_ĐOÁN",
                    "assertions": [],
                    "candidates": ["B18.2"],
                    "position": [0, len("Viêm gan virus C")],
                },
                {
                    "text": text,
                    "type": "CHẨN_ĐOÁN",
                    "assertions": [],
                    "candidates": ["B18.2", "B18.1"],
                    "position": [0, len(text)],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = audit_manual_gold_convention(input_dir, gold_dir, expected_count=1)

    assert report["blocking_count"] == 1
    assert report["by_kind"]["overlapping_entities"] == 1


def test_audit_requires_reviewed_decision_for_contextual_candidate_mapping(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    for document_id, candidate in (("1", "I21.9"), ("2", "I25.2")):
        text = "nhồi máu cơ tim"
        (input_dir / f"{document_id}.txt").write_text(text, encoding="utf-8")
        (gold_dir / f"{document_id}.json").write_text(
            json.dumps(
                [
                    {
                        "text": text,
                        "type": "CHẨN_ĐOÁN",
                        "assertions": [],
                        "candidates": [candidate],
                        "position": [0, len(text)],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    unresolved = audit_manual_gold_convention(input_dir, gold_dir, expected_count=2)
    assert unresolved["by_kind"] == {"candidate_mapping_conflict": 1}

    (gold_dir / "convention_decisions.jsonl").write_text(
        json.dumps(
            {
                "kind": "candidate_mapping_conflict",
                "entity_type": "CHẨN_ĐOÁN",
                "normalized_mention": "nhồi máu cơ tim",
                "decision": "allow",
                "reason": "The code depends on whether the context states an acute or old infarction.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = audit_manual_gold_convention(input_dir, gold_dir, expected_count=2)
    assert resolved["review_count"] == 0
    assert resolved["resolved_count"] == 1
