import json
from pathlib import Path
import zipfile

from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import (
    PHASE1_ENTITY_TYPE_ORDER,
    expand_repeated_phase1_mentions,
    load_phase1_output_source,
    merge_phase1_outputs,
    rank_phase1_source_strategies,
)


def test_load_phase1_output_source_accepts_root_level_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "predictions.zip"
    payload = [_row("ho", "TRIỆU_CHỨNG", 0)]
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("12.json", json.dumps(payload, ensure_ascii=False))

    loaded = load_phase1_output_source(zip_path)

    assert loaded == {"12": payload}


def test_merge_phase1_outputs_selects_source_by_type_and_empties_metadata() -> None:
    primary = {
        "12": [
            _row("tăng huyết áp", "CHẨN_ĐOÁN", 0, assertions=["isHistorical"], candidates=["I10"]),
            _row("ho", "TRIỆU_CHỨNG", 20),
        ]
    }
    secondary = {
        "12": [
            _row("sốt", "TRIỆU_CHỨNG", 30),
            _row("CRP", "TÊN_XÉT_NGHIỆM", 40),
        ]
    }
    strategies = {entity_type: "primary" for entity_type in PHASE1_ENTITY_TYPE_ORDER}
    strategies["TRIỆU_CHỨNG"] = "secondary"
    strategies["TÊN_XÉT_NGHIỆM"] = "secondary"

    merged = merge_phase1_outputs(primary, secondary, strategies)

    assert [row["text"] for row in merged["12"]] == ["tăng huyết áp", "sốt", "CRP"]
    assert merged["12"][0]["assertions"] == []
    assert merged["12"][0]["candidates"] == []


def test_merge_phase1_union_deduplicates_exact_entities() -> None:
    row = _row("ho", "TRIỆU_CHỨNG", 0)
    strategies = {entity_type: "primary" for entity_type in PHASE1_ENTITY_TYPE_ORDER}
    strategies["TRIỆU_CHỨNG"] = "union"

    merged = merge_phase1_outputs({"12": [row]}, {"12": [dict(row)]}, strategies)

    assert merged["12"] == [row]


def test_secondary_preferred_union_only_adds_nonoverlapping_primary_rows() -> None:
    primary = {
        "12": [
            _row("đau ngực", "TRIỆU_CHỨNG", 0),
            _row("sốt", "TRIỆU_CHỨNG", 20),
        ]
    }
    secondary = {"12": [_row("đau ngực dữ dội", "TRIỆU_CHỨNG", 0)]}
    strategies = {entity_type: "secondary" for entity_type in PHASE1_ENTITY_TYPE_ORDER}
    strategies["TRIỆU_CHỨNG"] = "secondary_preferred_union"

    merged = merge_phase1_outputs(primary, secondary, strategies)

    assert [row["text"] for row in merged["12"]] == ["đau ngực dữ dội", "sốt"]


def test_expand_repeated_mentions_recovers_same_line_occurrences() -> None:
    source_text = "Triệu chứng: ho, ho.\nKhông mở rộng ho ở dòng sau."
    rows = {"12": [_row("ho", "TRIỆU_CHỨNG", 13), _row("ho", "TRIỆU_CHỨNG", 13)]}

    expanded = expand_repeated_phase1_mentions(rows, {"12": source_text})

    assert [row["position"] for row in expanded["12"]] == [[13, 15], [17, 19]]


def test_strategy_search_ranks_on_train_and_reports_holdout() -> None:
    gold = {
        "11": [_row("ho", "TRIỆU_CHỨNG", 0)],
        "12": [_row("sốt", "TRIỆU_CHỨNG", 0)],
    }
    primary = {"11": [], "12": [_row("sốt", "TRIỆU_CHỨNG", 0)]}
    secondary = {"11": [_row("ho", "TRIỆU_CHỨNG", 0)], "12": []}

    ranked = rank_phase1_source_strategies(gold, primary, secondary)

    assert ranked[0]["source_by_type"]["TRIỆU_CHỨNG"] == "primary"
    assert ranked[0]["splits"]["train"]["metrics"]["score"] == 100.0
    assert ranked[0]["splits"]["holdout"]["metrics"]["score"] == 0.0


def _row(
    text: str,
    entity_type: str,
    start: int,
    *,
    assertions: list[str] | None = None,
    candidates: list[str] | None = None,
) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "assertions": assertions or [],
        "candidates": candidates or [],
        "position": [start, start + len(text)],
    }
