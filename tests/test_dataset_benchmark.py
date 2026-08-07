"""Contract tests for the public synthetic benchmark pilot."""

from pathlib import Path

from clingrounder.evaluation.dataset_benchmark import run_dataset_benchmark


def test_public_benchmark_writes_complete_artifact_bundle(tmp_path: Path) -> None:
    summary = run_dataset_benchmark(
        "benchmarks/vi_clinical_grounding_v1",
        "configs/benchmarks/vi_clinical_grounding_v1/full.yaml",
        tmp_path / "artifact",
    )

    assert summary["schema_version"] == "clingrounder.benchmark-summary.v1"
    assert summary["benchmark"]["status"] == "synthetic_pilot"
    assert summary["metrics"]["offset_validity"] == 1.0
    assert summary["error_count"] == 0
    expected = {
        "manifest.json",
        "summary.json",
        "report.md",
        "predictions.jsonl",
        "errors.json",
        "confusion-matrices.json",
        "runtime.json",
    }
    assert {path.name for path in (tmp_path / "artifact").iterdir()} == expected
