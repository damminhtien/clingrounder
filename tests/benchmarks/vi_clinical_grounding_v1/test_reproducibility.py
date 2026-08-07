"""Deterministic artifact checks for the product benchmark runner."""

import json
from pathlib import Path

from clingrounder.evaluation.dataset_benchmark import run_dataset_benchmark


def test_predictions_and_core_reports_are_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_dataset_benchmark(
        "benchmarks/vi_clinical_grounding_v1",
        "configs/benchmarks/vi_clinical_grounding_v1/full.yaml",
        first,
    )
    run_dataset_benchmark(
        "benchmarks/vi_clinical_grounding_v1",
        "configs/benchmarks/vi_clinical_grounding_v1/full.yaml",
        second,
    )

    for filename in ("manifest.json", "predictions.jsonl", "errors.json", "confusion-matrices.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    first_summary = json.loads((first / "summary.json").read_text())
    second_summary = json.loads((second / "summary.json").read_text())
    assert first_summary["metrics"] == second_summary["metrics"]
    assert first_summary["config_fingerprint"] == second_summary["config_fingerprint"]
    assert first_summary["terminology_fingerprint"] == second_summary["terminology_fingerprint"]

