from __future__ import annotations

from pathlib import Path

import pytest


_PRIVATE_PHASE1_TESTS = {
    "tests/test_manual_gold.py::test_validate_complete_manual_gold_batch",
    "tests/test_rule_ner.py::test_rule_ner_phase1_latency_under_100ms_per_note",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip tests that require the private Phase 1 corpus when it is unavailable.

    The repository intentionally ignores ``data/raw/*``. Public GitHub Actions
    runners therefore cannot execute corpus-wide validation or latency tests,
    while local environments with the complete 100-document corpus still run
    them normally.
    """

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
