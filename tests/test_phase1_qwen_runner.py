"""Runner helpers keep support evidence typed, offset-safe, and model-neutral."""

from __future__ import annotations

import pytest

from medical_kg_nlp.benchmarks.phase1.qwen_runner import (
    _adjudication_candidates,
    _proposals_to_rows,
    _rows_to_proposals,
)


def test_runner_round_trips_support_rows_without_assertion_or_candidates() -> None:
    text = "Bệnh nhân ho và tăng huyết áp"
    rows = [
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "position": [10, 12],
            "assertions": ["isNegated"],
            "candidates": [],
        },
        {
            "text": "tăng huyết áp",
            "type": "CHẨN_ĐOÁN",
            "position": [16, 29],
            "assertions": [],
            "candidates": ["I10"],
        },
    ]

    proposals = _rows_to_proposals(rows, text, source="rule")
    output = _proposals_to_rows(proposals, text)

    assert [row["text"] for row in output] == ["ho", "tăng huyết áp"]
    assert all(row["assertions"] == [] and row["candidates"] == [] for row in output)


def test_runner_rejects_support_offset_mismatch() -> None:
    with pytest.raises(ValueError, match="violates raw offsets"):
        _rows_to_proposals(
            (
                {
                    "text": "ho",
                    "type": "TRIỆU_CHỨNG",
                    "position": [0, 2],
                },
            ),
            "Không ho",
            source="xlmr",
        )


def test_adjudication_candidates_merge_exact_source_evidence() -> None:
    text = "ho"
    rule = _rows_to_proposals(
        ({"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2]},),
        text,
        source="rule",
    )
    qwen = _rows_to_proposals(
        ({"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2]},),
        text,
        source="qwen.recall",
    )

    candidates = _adjudication_candidates(
        {"rule": rule, "qwen.recall": qwen},
        text,
    )

    assert candidates[0].sources == ("qwen.recall", "rule")
