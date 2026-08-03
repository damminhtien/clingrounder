from __future__ import annotations

from pathlib import Path

import pytest


_PRIVATE_PHASE1_TESTS = {
    (
        "tests/benchmarks/phase1/test_manual_gold.py::"
        "test_validate_complete_manual_gold_batch"
    ),
    "tests/test_rule_ner.py::test_rule_ner_phase1_latency_under_100ms_per_note",
}
_PHASE1_BENCHMARK_PREFIX = "tests/benchmarks/phase1/"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify optional benchmark tests and skip unavailable private-corpus checks.

    Phase 1 tests live below one discoverable directory and run in nightly/full
    verification. The repository intentionally ignores ``data/raw/*``; checks
    requiring that corpus are skipped when its 100 source files are absent.
    """

    for item in items:
        if item.nodeid.startswith(_PHASE1_BENCHMARK_PREFIX):
            item.add_marker(pytest.mark.benchmark)

    input_dir = Path("data/raw/input")
    corpus_files = list(input_dir.glob("*.txt")) if input_dir.is_dir() else []
    if len(corpus_files) == 100:
        return

    skip_private_corpus = pytest.mark.skip(
        reason=(
            "requires the private Phase 1 corpus at data/raw/input "
            f"(expected 100 TXT files, found {len(corpus_files)})"
        )
    )
    for item in items:
        if item.nodeid in _PRIVATE_PHASE1_TESTS:
            item.add_marker(skip_private_corpus)
