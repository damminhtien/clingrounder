from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.manual_gold_candidates import (
    build_manual_gold_candidate_dictionary,
    write_manual_gold_candidate_dictionary,
)


def test_candidate_dictionary_uses_standard_rows_and_reviewed_aliases(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "1.json").write_text(
        json.dumps(
            [
                {
                    "text": "tăng huyết áp",
                    "type": "CHẨN_ĐOÁN",
                    "candidates": ["I10"],
                    "assertions": [],
                    "position": [0, 13],
                },
                {
                    "text": "aspirin 81 mg po daily",
                    "type": "THUỐC",
                    "candidates": ["243670"],
                    "assertions": [],
                    "position": [14, 36],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.jsonl"
    source.write_text(
        _jsonl(
            {
                "concept_id": "ICD10:I10",
                "code": "I10",
                "code_system": "ICD-10",
                "canonical_name": "Bệnh tăng huyết áp vô căn",
                "semantic_type": "DISEASE",
                "aliases": [],
                "source": "icd10_vn_tt06_2026",
            },
            {
                "concept_id": "RXNORM:243670",
                "code": "243670",
                "code_system": "RxNorm",
                "canonical_name": "aspirin 81 MG Oral Tablet",
                "semantic_type": "DRUG",
                "aliases": [],
                "source": "rxnorm_full_2026_07_06",
                "rxnorm_tty": "SCD",
            },
        ),
        encoding="utf-8",
    )

    rows, audit = build_manual_gold_candidate_dictionary(gold_dir, [source])

    assert audit["issue_count"] == 0
    assert audit["compiled_concept_count"] == 2
    aspirin = next(row for row in rows if row["code"] == "243670")
    assert aspirin["aliases"] == ["aspirin 81 mg po daily"]
    assert aspirin["manual_gold_document_support"] == 1

    output = tmp_path / "reviewed.jsonl"
    assert write_manual_gold_candidate_dictionary(rows, output) == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_candidate_dictionary_reports_missing_or_wrong_standard_code(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "1.json").write_text(
        json.dumps(
            [
                {
                    "text": "thuốc thử",
                    "type": "THUỐC",
                    "candidates": ["123"],
                    "assertions": [],
                    "position": [0, 9],
                },
                {
                    "text": "bệnh thiếu",
                    "type": "CHẨN_ĐOÁN",
                    "candidates": ["X00"],
                    "assertions": [],
                    "position": [10, 20],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.jsonl"
    source.write_text(
        _jsonl(
            {
                "concept_id": "RXNORM:123",
                "code": "123",
                "code_system": "RxNorm",
                "canonical_name": "wrong type",
                "semantic_type": "DISEASE",
                "aliases": [],
            }
        ),
        encoding="utf-8",
    )

    rows, audit = build_manual_gold_candidate_dictionary(gold_dir, [source])

    assert rows == []
    assert {issue["kind"] for issue in audit["issues"]} == {
        "missing_standard_code",
        "standard_semantic_type_mismatch",
    }


def _jsonl(*rows: dict[str, object]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
