"""Contracts for explicit Phase 1 assertion-overlay resources."""

from __future__ import annotations

from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.assertion_overlays import (
    load_phase1_assertion_overlays,
)


def test_phase1_assertion_overlays_are_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "overlays.jsonl"
    path.write_text(
        '{"assertion":"isHistorical","entity_types":["DRUG"],'
        '"left_regex":"allergy\\\\s*$"}\n',
        encoding="utf-8",
    )

    assert load_phase1_assertion_overlays(None) == ()
    overlays = load_phase1_assertion_overlays(path)
    assert len(overlays) == 1
    assert overlays[0].assertion == "isHistorical"
    assert overlays[0].matches("allergy aspirin", (8, 15), entity_type="DRUG")


def test_phase1_assertion_overlay_path_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="assertion overlay does not exist"):
        load_phase1_assertion_overlays(tmp_path / "missing.jsonl")
