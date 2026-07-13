from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.evaluation.phase1_selective_overlays import (
    apply_selective_assertions,
    apply_selective_candidates,
    compile_reviewed_candidate_registry,
    write_reviewed_candidate_map,
    validate_probe_isolation,
)


def test_assertion_regimes_use_scoped_history_negation_and_family_without_changing_entities() -> None:
    text = (
        "Tiền sử bệnh:\nTăng huyết áp\n"
        "Triệu chứng hiện tại:\nKhông có sốt, ho nhưng có đau ngực.\n"
        "Tiền sử gia đình:\nMẹ có đái tháo đường"
    )
    rows = {
        "1": [
            _row("Tăng huyết áp", "CHẨN_ĐOÁN", text.index("Tăng huyết áp")),
            _row("sốt", "TRIỆU_CHỨNG", text.index("sốt")),
            _row("ho", "TRIỆU_CHỨNG", text.index("ho nhưng")),
            _row("đau ngực", "TRIỆU_CHỨNG", text.index("đau ngực")),
            _row("đái tháo đường", "CHẨN_ĐOÁN", text.index("đái tháo đường")),
        ]
    }

    output, decisions, _ = apply_selective_assertions(
        rows,
        {"1": text},
        regimes=("history", "negation", "family"),
    )

    assert [row["assertions"] for row in output["1"]] == [
        ["isHistorical"],
        ["isNegated"],
        ["isNegated"],
        [],
        ["isFamily"],
    ]
    assert all("rule_id" in decision for decision in decisions)
    assert validate_probe_isolation(rows, output, module="assertion") == []


def test_negation_exceptions_do_not_invert_positive_inability_or_disease_modifier() -> None:
    text = "Bệnh nhân không thể đứng và đau thắt ngực không ổn định."
    rows = {
        "1": [
            _row("đứng", "TRIỆU_CHỨNG", text.index("đứng")),
            _row(
                "đau thắt ngực không ổn định",
                "CHẨN_ĐOÁN",
                text.index("đau thắt ngực"),
            ),
        ]
    }

    output, _, _ = apply_selective_assertions(
        rows,
        {"1": text},
        regimes=("negation",),
    )

    assert [row["assertions"] for row in output["1"]] == [[], []]


def test_history_scope_stops_at_current_preadmission_state_and_ignores_recent_onset() -> None:
    text = (
        "Tiền sử bệnh:\nTăng huyết áp\n"
        "Tình trạng ngay trước khi nhập viện:\nho\n"
        "Bệnh sử hiện tại: cách đây 1 tuần khó thở. "
        "Cách đây vài năm thoái hóa khớp."
    )
    rows = {
        "1": [
            _row("Tăng huyết áp", "CHẨN_ĐOÁN", text.index("Tăng huyết áp")),
            _row("ho", "TRIỆU_CHỨNG", text.index("ho\n")),
            _row("khó thở", "TRIỆU_CHỨNG", text.index("khó thở")),
            _row("thoái hóa khớp", "CHẨN_ĐOÁN", text.index("thoái hóa khớp")),
        ]
    }

    output, _, _ = apply_selective_assertions(
        rows,
        {"1": text},
        regimes=("history",),
    )

    assert [row["assertions"] for row in output["1"]] == [
        ["isHistorical"],
        [],
        [],
        ["isHistorical"],
    ]


def test_history_policy_abstains_on_symptoms_even_inside_history_section() -> None:
    text = "Tiền sử bệnh:\nđau đầu\nTăng huyết áp"
    rows = {
        "1": [
            _row("đau đầu", "TRIỆU_CHỨNG", text.index("đau đầu")),
            _row("Tăng huyết áp", "CHẨN_ĐOÁN", text.index("Tăng huyết áp")),
        ]
    }

    output, _, _ = apply_selective_assertions(
        rows,
        {"1": text},
        regimes=("history",),
    )

    assert [row["assertions"] for row in output["1"]] == [[], ["isHistorical"]]


def test_assertion_extension_preserves_winning_baseline_labels() -> None:
    text = "Tiền sử gia đình:\nMẹ có đái tháo đường\nTriệu chứng hiện tại:\nKhông có sốt"
    rows = {
        "1": [
            {
                **_row("đái tháo đường", "CHẨN_ĐOÁN", text.index("đái tháo đường")),
                "assertions": ["isHistorical"],
            },
            {
                **_row("sốt", "TRIỆU_CHỨNG", text.index("sốt")),
                "assertions": ["isNegated"],
            },
        ]
    }

    output, _, _ = apply_selective_assertions(
        rows,
        {"1": text},
        regimes=("family",),
        preserve_existing=True,
    )

    assert [row["assertions"] for row in output["1"]] == [
        ["isHistorical", "isFamily"],
        ["isNegated"],
    ]


def test_candidate_registry_uses_train_review_tt06_and_source_consensus(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.jsonl"
    concepts = [
        _concept(
            "I10",
            "ICD-10",
            canonical_name="tăng huyết áp",
            source_ids=["icd10_vn_tt06_2026"],
        ),
        _concept(
            "4603",
            "RxNorm",
            canonical_name="lasix",
            tty="IN",
            source_ids=["rxnorm_full_2026_07_06"],
        ),
        _concept(
            "999",
            "RxNorm",
            canonical_name="drug 20mg tablet",
            tty="SCD",
            source_ids=["rxnorm_full_2026_07_06"],
        ),
    ]
    dictionary.write_text(
        "".join(json.dumps(row) + "\n" for row in concepts),
        encoding="utf-8",
    )
    gold = {
        "1": [
            _row("tăng huyết áp", "CHẨN_ĐOÁN", 0, candidates=["I10"]),
            _row("lasix", "THUỐC", 20, candidates=["4603"]),
            _row("drug 20mg tablet", "THUỐC", 40, candidates=["999"]),
        ]
    }
    registry, audit = compile_reviewed_candidate_registry(gold, dictionary, split="all")
    reviewed_path = tmp_path / "reviewed_candidate_map.jsonl"
    reviewed_rows = write_reviewed_candidate_map(registry, reviewed_path)
    rows = {
        "1": [
            _row("tăng huyết áp", "CHẨN_ĐOÁN", 0),
            _row("lasix", "THUỐC", 20),
            _row("drug 20mg tablet", "THUỐC", 40),
        ]
    }
    consensus = {
        ("1", 0, len("tăng huyết áp"), "CHẨN_ĐOÁN"),
        ("1", 20, 20 + len("lasix"), "THUỐC"),
        ("1", 40, 40 + len("drug 20mg tablet"), "THUỐC"),
    }

    icd, _, _ = apply_selective_candidates(rows, registry, regime="icd", consensus_keys=consensus)
    ingredient, _, _ = apply_selective_candidates(
        rows, registry, regime="rxnorm_ingredient", consensus_keys=consensus
    )
    clinical, _, _ = apply_selective_candidates(
        rows, registry, regime="rxnorm_clinical_drug", consensus_keys=consensus
    )

    assert [row["candidates"] for row in icd["1"]] == [["I10"], [], []]
    assert [row["candidates"] for row in ingredient["1"]] == [[], ["4603"], []]
    assert [row["candidates"] for row in clinical["1"]] == [[], [], ["999"]]
    assert audit["compiled_rule_count"] == 3
    assert len(reviewed_rows) == 3
    assert all(row["review_status"] == "reviewed" for row in reviewed_rows)
    assert all(row["code_system"] in {"ICD-10", "RxNorm"} for row in reviewed_rows)
    assert reviewed_path.read_text(encoding="utf-8").endswith("\n")
    assert validate_probe_isolation(rows, clinical, module="candidate") == []


def test_candidate_regimes_can_accumulate_in_full_diagnostic_mode(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.jsonl"
    dictionary.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                _concept(
                    "I10",
                    "ICD-10",
                    canonical_name="tăng huyết áp",
                    source_ids=["icd10_vn_tt06_2026"],
                ),
                _concept(
                    "4603",
                    "RxNorm",
                    canonical_name="lasix",
                    tty="IN",
                    source_ids=["rxnorm_full_2026_07_06"],
                ),
            )
        ),
        encoding="utf-8",
    )
    gold = {
        "1": [
            _row("tăng huyết áp", "CHẨN_ĐOÁN", 0, candidates=["I10"]),
            _row("lasix", "THUỐC", 20, candidates=["4603"]),
        ]
    }
    registry, _ = compile_reviewed_candidate_registry(gold, dictionary, split="all")
    rows = {"1": [_row("tăng huyết áp", "CHẨN_ĐOÁN", 0), _row("lasix", "THUỐC", 20)]}
    consensus = {
        ("1", 0, len("tăng huyết áp"), "CHẨN_ĐOÁN"),
        ("1", 20, 20 + len("lasix"), "THUỐC"),
    }

    icd, _, _ = apply_selective_candidates(
        rows,
        registry,
        regime="icd",
        consensus_keys=consensus,
        preserve_existing=True,
    )
    combined, _, _ = apply_selective_candidates(
        icd,
        registry,
        regime="rxnorm_ingredient",
        consensus_keys=consensus,
        preserve_existing=True,
    )

    assert [row["candidates"] for row in combined["1"]] == [["I10"], ["4603"]]


def test_candidate_overlay_requires_exact_two_source_consensus(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.jsonl"
    dictionary.write_text(
        json.dumps(
            _concept(
                "I10",
                "ICD-10",
                canonical_name="tăng huyết áp",
                source_ids=["icd10_vn_tt06_2026"],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    gold = {"1": [_row("tăng huyết áp", "CHẨN_ĐOÁN", 0, candidates=["I10"])]}
    registry, _ = compile_reviewed_candidate_registry(gold, dictionary, split="all")
    rows = {"1": [_row("tăng huyết áp", "CHẨN_ĐOÁN", 0)]}

    output, decisions, counts = apply_selective_candidates(
        rows,
        registry,
        regime="icd",
        consensus_keys=set(),
    )

    assert output["1"][0]["candidates"] == []
    assert decisions == []
    assert counts["blocked_without_two_source_consensus"] == 1


def test_candidate_registry_rejects_ambiguous_and_dictionary_disagreeing_mentions(
    tmp_path: Path,
) -> None:
    dictionary = tmp_path / "dictionary.jsonl"
    concepts = [
        _concept(
            "R68.0",
            "ICD-10",
            canonical_name="hạ thân nhiệt",
            source_ids=["icd10_vn_tt06_2026"],
        ),
        _concept(
            "T68",
            "ICD-10",
            canonical_name="hạ thân nhiệt do môi trường",
            aliases=["hạ thân nhiệt"],
            source_ids=["icd10_vn_tt06_2026"],
        ),
        _concept(
            "K65.0",
            "ICD-10",
            canonical_name="áp xe phúc mạc",
            source_ids=["icd10_vn_tt06_2026"],
        ),
        _concept(
            "L02.9",
            "ICD-10",
            canonical_name="áp xe",
            source_ids=["icd10_vn_tt06_2026"],
        ),
    ]
    dictionary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in concepts),
        encoding="utf-8",
    )
    gold = {
        "1": [
            _row("hạ thân nhiệt", "CHẨN_ĐOÁN", 0, candidates=["R68.0"]),
            _row("áp xe", "CHẨN_ĐOÁN", 20, candidates=["K65.0"]),
        ]
    }

    registry, audit = compile_reviewed_candidate_registry(gold, dictionary, split="all")

    assert registry.rules == ()
    assert audit["compiled_rule_count"] == 0
    assert audit["rejected_counts"] == {
        "not_exact_unique_dictionary_match": 1,
        "reviewed_code_disagrees_with_exact_dictionary": 1,
    }


def _row(
    text: str,
    entity_type: str,
    start: int,
    *,
    candidates: list[str] | None = None,
) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "candidates": candidates or [],
        "position": [start, start + len(text)],
    }


def _concept(
    code: str,
    code_system: str,
    *,
    canonical_name: str,
    tty: str | None = None,
    source_ids: list[str],
    aliases: list[str] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "concept_id": f"{code_system}:{code}",
        "code": code,
        "code_system": code_system,
        "canonical_name": canonical_name,
        "semantic_type": "DISEASE" if code_system == "ICD-10" else "DRUG",
        "aliases": aliases or [],
        "source_ids": source_ids,
    }
    if tty:
        row["rxnorm_tty"] = tty
    return row
