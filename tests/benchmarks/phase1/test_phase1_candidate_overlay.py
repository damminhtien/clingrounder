import json
from pathlib import Path

from clingrounder.benchmarks.phase1.phase1_candidate_overlay import (
    Phase1CandidateIndex,
    Phase1CandidateOverlayConfig,
    apply_phase1_candidate_overlay,
    candidate_ablation_passes,
)


def test_candidate_index_requires_tt06_and_unique_exact_alias(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.jsonl"
    _write_dictionary(
        dictionary,
        [
            _concept("I10", "ICD-10", "Tăng huyết áp", source_ids=["icd10_vn_tt06_2026"]),
            _concept("I11", "ICD-10", "Bệnh tim tăng huyết áp", aliases=["THA"], source_ids=["who"]),
            _concept("I12", "ICD-10", "Khác", aliases=["mơ hồ"], source_ids=["icd10_vn_tt06_2026"]),
            _concept("I13", "ICD-10", "Khác nữa", aliases=["mơ hồ"], source_ids=["icd10_vn_tt06_2026"]),
        ],
    )

    index = Phase1CandidateIndex.from_jsonl(dictionary)

    assert index.icd_exact["tăng huyết áp"] == "I10"
    assert "tha" not in index.icd_exact
    assert "mơ hồ" not in index.icd_exact


def test_rxnorm_exact_and_longest_require_unique_code(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.jsonl"
    _write_dictionary(
        dictionary,
        [
            _concept("4603", "RxNorm", "Furosemide", aliases=["Lasix"]),
            _concept("1", "RxNorm", "Drug one", aliases=["Combo"]),
            _concept("2", "RxNorm", "Drug two", aliases=["Combo"]),
        ],
    )
    index = Phase1CandidateIndex.from_jsonl(dictionary)

    assert index.rxnorm_exact["lasix"] == "4603"
    assert "combo" not in index.rxnorm_exact
    assert index.longest_unique_rxnorm_code("80mg po Lasix IV") == "4603"
    assert index.longest_unique_rxnorm_code("Combo 20mg") is None


def test_candidate_overlay_assigns_at_most_one_code_by_type(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.jsonl"
    _write_dictionary(
        dictionary,
        [
            _concept("I10", "ICD-10", "Tăng huyết áp", source_ids=["icd10_vn_tt06_2026"]),
            _concept("4603", "RxNorm", "Furosemide", aliases=["Lasix"]),
        ],
    )
    index = Phase1CandidateIndex.from_jsonl(dictionary)
    rows = {
        "1": [
            _row("Tăng huyết áp", "CHẨN_ĐOÁN"),
            _row("80mg Lasix IV", "THUỐC"),
            _row("ho", "TRIỆU_CHỨNG"),
        ]
    }

    overlaid, counts = apply_phase1_candidate_overlay(
        rows,
        index,
        Phase1CandidateOverlayConfig(icd_exact=True, rxnorm_exact=True, rxnorm_longest=True),
    )

    assert [row["candidates"] for row in overlaid["1"]] == [["I10"], ["4603"], []]
    assert counts == {"assigned_total": 2, "icd_exact": 1, "rxnorm_longest": 1}


def test_candidate_gate_requires_both_total_and_candidate_gain() -> None:
    baseline = {"score": 50.0, "candidates_score": 0.4}

    assert candidate_ablation_passes(baseline, {"score": 50.1, "candidates_score": 0.41})
    assert not candidate_ablation_passes(baseline, {"score": 50.1, "candidates_score": 0.4})
    assert not candidate_ablation_passes(baseline, {"score": 49.9, "candidates_score": 0.41})


def _concept(
    code: str,
    code_system: str,
    canonical_name: str,
    *,
    aliases: list[str] | None = None,
    source_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "concept_id": f"{code_system}:{code}",
        "code": code,
        "code_system": code_system,
        "canonical_name": canonical_name,
        "semantic_type": "DISEASE" if code_system == "ICD-10" else "DRUG",
        "aliases": aliases or [],
        "source_ids": source_ids or [],
    }


def _write_dictionary(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(text: str, entity_type: str) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "candidates": ["stale"],
        "position": [0, len(text)],
    }
