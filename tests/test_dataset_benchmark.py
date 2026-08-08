"""Contract tests for the public synthetic benchmark pilot."""

from pathlib import Path

from clingrounder.evaluation.dataset_benchmark import (
    run_dataset_benchmark,
    run_dataset_benchmark_suite,
)


def test_public_benchmark_writes_complete_artifact_bundle(tmp_path: Path) -> None:
    summary = run_dataset_benchmark(
        "benchmarks/vi_clinical_grounding_v1",
        "configs/benchmarks/vi_clinical_grounding_v1/full.yaml",
        tmp_path / "artifact",
    )

    assert summary["schema_version"] == "clingrounder.benchmark-summary.v1"
    assert summary["benchmark"]["status"] == "synthetic_pilot"
    assert summary["metrics"]["offset_validity"] == 1.0
    assert summary["metrics"]["entity_overlap_micro_f1"] == 1.0
    assert summary["metrics"]["assertion_accuracy"] == 1.0
    assert summary["metrics"]["linking_mrr"] == 1.0
    assert summary["git_commit"]
    assert summary["performance"]["peak_rss_mb"] >= 0
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


def test_public_benchmark_suite_writes_named_runs_and_index(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    suite = run_dataset_benchmark_suite(
        "benchmarks/vi_clinical_grounding_v1",
        {
            "full": "configs/benchmarks/vi_clinical_grounding_v1/full.yaml",
            "exact": "configs/benchmarks/vi_clinical_grounding_v1/exact.yaml",
        },
        output,
    )

    assert suite["schema_version"] == "clingrounder.benchmark-suite.v1"
    assert list(suite["runs"]) == ["exact", "full"]
    assert (output / "exact" / "summary.json").is_file()
    assert (output / "full" / "predictions.jsonl").is_file()
    assert (output / "suite.json").is_file()
    assert "| exact |" in (output / "report.md").read_text(encoding="utf-8")


def test_benchmark_suite_rejects_path_traversal_names(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="Invalid benchmark suite config name"):
        run_dataset_benchmark_suite(
            "benchmarks/vi_clinical_grounding_v1",
            {"../escape": "configs/benchmarks/vi_clinical_grounding_v1/exact.yaml"},
            tmp_path / "suite",
        )
