import json
from pathlib import Path

from clingrounder.dictionaries.alias_mining import mine_vietnamese_alias_candidates, write_alias_mining_outputs


def test_mine_vietnamese_alias_candidates_proposes_standard_alias_missing_runtime(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text(
        "Bệnh nhân có đái tháo đường típ 1. Uống thuốc lorazepam 1mg.",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.jsonl"
    _write_jsonl(
        runtime,
        [
            {
                "concept_id": "ICD10:E11",
                "code": "E11",
                "code_system": "ICD-10",
                "canonical_name": "Đái tháo đường típ 2",
                "semantic_type": "DISEASE",
                "aliases": ["đái tháo đường"],
                "source": "seed",
            }
        ],
    )
    standard = tmp_path / "standard.jsonl"
    _write_jsonl(
        standard,
        [
            {
                "concept_id": "ICD10:E10",
                "code": "E10",
                "code_system": "ICD-10",
                "canonical_name": "Đái tháo đường típ 1",
                "official_name_vi": "Đái tháo đường típ 1",
                "semantic_type": "DISEASE",
                "source_ids": ["icd10_vn_tt06_2026"],
            },
            {
                "concept_id": "RXNORM:6470",
                "code": "6470",
                "code_system": "RxNorm",
                "canonical_name": "lorazepam",
                "semantic_type": "DRUG",
                "source_ids": ["rxnorm_prescribable_2026_06_01"],
            },
        ],
    )

    candidates = mine_vietnamese_alias_candidates(
        input_dir=input_dir,
        runtime_dictionary_path=runtime,
        standard_dictionary_paths=[standard],
    )

    by_alias = {row.get("normalized_alias"): row for row in candidates}
    assert by_alias["đái tháo đường típ 1"]["target_concept_id"] == "ICD10:E10"
    assert by_alias["lorazepam"]["target_concept_id"] == "RXNORM:6470"
    assert all(row["recommended_action"].startswith("review") for row in candidates)


def test_write_alias_mining_outputs_writes_jsonl_and_markdown(tmp_path: Path) -> None:
    candidates = [
        {
            "proposal_type": "unknown_phrase",
            "priority": 10,
            "term": "men gan",
            "count": 3,
            "recommended_action": "review_classify_as_alias_section_or_ignore",
        }
    ]

    write_alias_mining_outputs(candidates, tmp_path)

    assert (tmp_path / "alias_candidates.jsonl").read_text(encoding="utf-8").strip()
    assert "men gan" in (tmp_path / "alias_candidates.md").read_text(encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
