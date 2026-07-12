from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.evaluation.phase1_selective_overlays import (
    apply_selective_assertions,
    apply_selective_candidates,
    compile_reviewed_candidate_registry,
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
        "Cách đây vài năm đau đầu."
    )
    rows = {
        "1": [
            _row("Tăng huyết áp", "CHẨN_ĐOÁN", text.index("Tăng huyết áp")),
            _row("ho", "TRIỆU_CHỨNG", text.index("ho\n")),
            _row("khó thở", "TRIỆU_CHỨNG", text.index("khó thở")),
            _row("đau đầu", "TRIỆU_CHỨNG", text.index("đau đầu")),
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


def test_candidate_registry_uses_train_review_tt06_and_source_consensus(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.jsonl"
    concepts = [
        _concept("I10", "ICD-10", source_ids=["icd10_vn_tt06_2026"]),
        _concept("4603", "RxNorm", tty="IN", source_ids=["rxnorm_full_2026_07_06"]),
        _concept("999", "RxNorm", tty="SCD", source_ids=["rxnorm_full_2026_07_06"]),
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
    assert validate_probe_isolation(rows, clinical, module="candidate") == []


def test_candidate_overlay_requires_exact_two_source_consensus(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.jsonl"
    dictionary.write_text(
        json.dumps(_concept("I10", "ICD-10", source_ids=["icd10_vn_tt06_2026"])) + "\n",
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
    tty: str | None = None,
    source_ids: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "code": code,
        "code_system": code_system,
        "source_ids": source_ids,
    }
    if tty:
        row["rxnorm_tty"] = tty
    return row
