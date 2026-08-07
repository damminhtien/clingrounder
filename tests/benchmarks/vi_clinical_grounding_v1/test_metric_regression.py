"""Protected metric checks for the public benchmark pilot."""

from pathlib import Path

from clingrounder.evaluation.dataset_benchmark import run_dataset_benchmark


def test_full_profile_preserves_exact_baseline_metrics(tmp_path: Path) -> None:
    exact = run_dataset_benchmark(
        "benchmarks/vi_clinical_grounding_v1",
        "configs/benchmarks/vi_clinical_grounding_v1/exact.yaml",
        tmp_path / "exact",
    )
    full = run_dataset_benchmark(
        "benchmarks/vi_clinical_grounding_v1",
        "configs/benchmarks/vi_clinical_grounding_v1/full.yaml",
        tmp_path / "full",
    )

    exact_metrics = exact["metrics"]
    full_metrics = full["metrics"]
    assert full["error_count"] == 0
    assert full_metrics["offset_validity"] == 1.0
    assert full_metrics["entity_exact_micro_f1"] >= exact_metrics["entity_exact_micro_f1"]
    assert full_metrics["assertion_positive_macro_f1"] >= exact_metrics["assertion_positive_macro_f1"]
    assert full_metrics["linking_recall_at_5"] >= exact_metrics["linking_recall_at_5"]

