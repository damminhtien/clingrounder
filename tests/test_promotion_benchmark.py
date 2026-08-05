"""Promotion benchmark contracts and comparison gates."""

from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.evaluation.promotion_benchmark import (
    _load_inputs,
    compare_promotion_benchmarks,
)


def test_fixture_is_valid_jsonl_and_covers_required_slices() -> None:
    path = Path("data/benchmarks/promotion/runtime_documents.jsonl")
    rows = _load_inputs(path)
    slices = {row.metadata["slice"] for row in rows}
    assert {
        "short",
        "long_note",
        "medication_list",
        "lab_table",
        "negation_family",
        "noisy_vietnamese_english",
        "repeated_entities",
        "many_candidate_mentions",
    } <= slices
    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_benchmark_commands_are_parseable() -> None:
    parser = build_parser()
    run = parser.parse_args(
        [
            "benchmark",
            "runtime",
            "run",
            "--input",
            "input.jsonl",
            "--config",
            "profile.yaml",
            "--output",
            "report.json",
        ]
    )
    compare = parser.parse_args(["benchmark", "compare", "base.json", "candidate.json"])
    assert run.handler == "benchmark_runtime_run"
    assert compare.handler == "benchmark_compare"


def _report(*, candidate_recall: float, assertion_recall: float | None = 1.0) -> dict:
    return {
        "environment": {"commit": "test"},
        "correctness": {
            "offset_validity": 1.0,
            "invalid_assigned_code_rate": 0.0,
            "invalid_relation_rate": 0.0,
            "deterministic_output": True,
            "candidate_ordering_stable": True,
            "candidate_recall_at_k": {"1": candidate_recall, "5": candidate_recall, "20": candidate_recall},
            "assertion_positive_recall": assertion_recall,
        },
        "performance": {
            "document_latency_ms": {"p50": 10.0},
            "peak_rss_bytes": 100,
        },
    }


def test_compare_rejects_missing_protected_metric() -> None:
    result = compare_promotion_benchmarks(
        _report(candidate_recall=0.8, assertion_recall=None),
        _report(candidate_recall=0.8, assertion_recall=None),
    )
    assert result["promote"] is False
    assert result["gates"]["assertion_positive_recall"] is None


def test_compare_accepts_equal_metrics_within_tolerance() -> None:
    result = compare_promotion_benchmarks(_report(candidate_recall=0.8), _report(candidate_recall=0.8))
    assert result["promote"] is True
