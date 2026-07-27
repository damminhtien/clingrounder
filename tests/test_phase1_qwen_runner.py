"""Runner helpers keep support evidence typed, offset-safe, and model-neutral."""

from __future__ import annotations

from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.qwen_runner import (
    Phase1QwenProposalRunConfig,
    _adjudication_candidates,
    _merge_review_rows,
    _proposals_to_rows,
    _rows_to_review_entities,
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


def test_review_rows_preserve_baseline_metadata_and_reject_overlap() -> None:
    baseline = [
        {
            "text": "aspirin 81 mg",
            "type": "THUỐC",
            "position": [0, 13],
            "assertions": ["isHistorical"],
            "candidates": ["243670"],
        }
    ]
    additions = [
        {
            "text": "aspirin",
            "type": "THUỐC",
            "position": [0, 7],
            "assertions": [],
            "candidates": [],
        },
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "position": [18, 20],
            "assertions": [],
            "candidates": [],
        },
    ]

    reviewed, rejected = _merge_review_rows(baseline, additions)

    assert rejected == 1
    assert reviewed == [
        baseline[0],
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "position": [18, 20],
            "assertions": [],
            "candidates": [],
        },
    ]


def test_review_entities_keep_every_raw_occurrence() -> None:
    text = "ho rồi ho"
    entities = _rows_to_review_entities(
        (
            {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2]},
            {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [7, 9]},
        ),
        text,
        source="baseline",
    )

    assert [entity.span for entity in entities] == [(0, 2), (7, 9)]


def test_review_only_config_requires_a_complete_source(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires a review source"):
        Phase1QwenProposalRunConfig(
            documents_path=documents,
            expected_source_archive_sha256="a" * 64,
            output_dir=tmp_path / "output",
            review_only=True,
        )
