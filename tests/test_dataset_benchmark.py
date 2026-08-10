"""Contract tests for the public synthetic benchmark pilot."""

import json
from pathlib import Path

import pytest

from clingrounder.evaluation.dataset_benchmark import (
    BenchmarkExample,
    _validate_predictions,
    compare_dataset_benchmarks,
    run_dataset_benchmark,
    run_dataset_benchmark_suite,
    verify_dataset_benchmark_reference,
)
from clingrounder.schema.annotation import EntityAnnotation
from clingrounder.schema.output import ClinicalPrediction
from clingrounder.schema.types import CodeSystem, EntityType


class _EmptyTerminology:
    """Minimal membership port used to prove fabricated codes fail the benchmark gate."""

    def contains(self, code_system: CodeSystem, code: str) -> bool:
        del code_system, code
        return False


def _promotion_summary(
    *,
    entity_f1: float,
    linking_recall: float = 0.8,
    assertion_f1: float = 0.7,
    p95_ms: float = 100.0,
) -> dict[str, object]:
    return {
        "metrics": {
            "entity_exact_micro_f1": entity_f1,
            "linking_recall_at_5": linking_recall,
            "assertion_positive_macro_f1": assertion_f1,
            "offset_validity": 1.0,
            "invalid_assigned_code_rate": 0.0,
            "invalid_relation_rate": 0.0,
            "validation_error_count": 0,
        },
        "performance": {"document_latency_ms": {"p95": p95_ms}},
    }


def test_dataset_promotion_gate_requires_primary_gain_and_protected_metrics() -> None:
    policy = {
        "primary": {
            "metric": "entity_exact_micro_f1",
            "minimum_improvement": 0.005,
        },
        "protected": {
            "linking_recall_at_5": {"maximum_regression": 0.003},
            "assertion_positive_macro_f1": {"maximum_regression": 0.003},
            "p95_ms": {"maximum_regression_ratio": 0.10},
        },
    }

    promoted = compare_dataset_benchmarks(
        _promotion_summary(entity_f1=0.80),
        _promotion_summary(entity_f1=0.81, linking_recall=0.798, p95_ms=105.0),
        policy,
    )
    assert promoted["promote"] is True
    assert promoted["primary"]["delta"] == pytest.approx(0.01)
    wrapped_policy_result = compare_dataset_benchmarks(
        _promotion_summary(entity_f1=0.80),
        _promotion_summary(entity_f1=0.81, linking_recall=0.798, p95_ms=105.0),
        {"policy": policy},
    )
    assert wrapped_policy_result["promote"] is True

    rejected = compare_dataset_benchmarks(
        _promotion_summary(entity_f1=0.80),
        _promotion_summary(entity_f1=0.81, linking_recall=0.79, p95_ms=105.0),
        policy,
    )
    assert rejected["promote"] is False
    assert rejected["protected"]["linking_recall_at_5"]["passed"] is False


def test_public_benchmark_writes_complete_artifact_bundle(tmp_path: Path) -> None:
    summary = run_dataset_benchmark(
        "benchmarks/vi_clinical_grounding_v1",
        "configs/benchmarks/vi_clinical_grounding_v1/full.yaml",
        tmp_path / "artifact",
    )

    assert summary["schema_version"] == "clingrounder.benchmark-summary.v1"
    assert summary["benchmark"]["status"] == "synthetic_pilot"
    assert len(summary["benchmark_manifest_sha256"]) == 64
    assert len(summary["input_sha256"]) == 64
    assert len(summary["config_source_sha256"]) == 64
    assert summary["metrics"]["offset_validity"] == 1.0
    assert summary["metrics"]["entity_overlap_micro_f1"] == 1.0
    assert summary["metrics"]["assertion_accuracy"] == 1.0
    assert summary["metrics"]["linking_mrr"] == 0.875
    assert summary["metrics"]["linkable_gold_count"] == 8
    assert summary["metrics"]["assigned_prediction_count"] == 7
    assert summary["metrics"]["assignment_coverage"] == 7 / 9
    assert summary["metrics"]["relation_gold_count"] == 1
    assert summary["metrics"]["relation_predicted_count"] == 1
    assert summary["metrics"]["relation_micro_f1"] == 1.0
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


def test_linking_metrics_count_a_missing_gold_entity_as_a_recall_miss() -> None:
    from clingrounder.evaluation.dataset_benchmark import _score

    example = BenchmarkExample(
        document_id="missing-link",
        text="ho metformin",
        metadata={},
        entities=(
            {
                "id": "g1",
                "span": [0, 2],
                "text": "ho",
                "type": "SYMPTOM",
                "assertion": "PRESENT",
                "code_system": "LOCAL",
                "code": "SYMPTOM_COUGH",
            },
        ),
        relations=(),
    )

    metrics, _ = _score([example], {})

    assert metrics["linkable_gold_count"] == 1
    assert metrics["linking_recall_at_5"] == 0.0
    assert metrics["assignment_coverage"] == 0.0


def test_benchmark_validation_reports_offset_and_membership_failures() -> None:
    prediction = ClinicalPrediction.from_text(
        "invalid",
        "ho",
        [
            EntityAnnotation(
                id="e1",
                span=(0, 2),
                text="sốt",
                normalized_text="sốt",
                type=EntityType.SYMPTOM,
                code_system=CodeSystem.LOCAL,
                code="fabricated",
            )
        ],
        [],
        "test",
    )
    example = BenchmarkExample(
        document_id="invalid",
        text="ho",
        metadata={},
        entities=(),
        relations=(),
    )

    metrics = _validate_predictions(
        [example],
        {prediction.document_id: prediction},
        _EmptyTerminology(),
    )

    assert metrics["offset_validity"] == 0.0
    assert metrics["invalid_assigned_code_rate"] == 1.0
    assert metrics["validation_error_count"] == 2
    assert metrics["validation_error_kinds"] == {
        "offset": 1,
        "unknown_dictionary_code": 1,
    }


def test_benchmark_validation_fails_closed_for_missing_prediction() -> None:
    example = BenchmarkExample(
        document_id="missing",
        text="ho",
        metadata={},
        entities=(),
        relations=(),
    )

    metrics = _validate_predictions([example], {}, _EmptyTerminology())

    assert metrics["offset_validity"] == 0.0
    assert metrics["missing_prediction_count"] == 1


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
    assert len(suite["benchmark_manifest_sha256"]) == 64
    assert len(suite["input_sha256"]) == 64
    assert len(suite["runs"]["exact"]["profile_sha256"]) == 64
    assert len(suite["runs"]["exact"]["config_source_sha256"]) == 64
    assert len(suite["runs"]["exact"]["terminology_fingerprint"]) == 64
    assert (output / "exact" / "summary.json").is_file()
    assert (output / "full" / "predictions.jsonl").is_file()
    assert (output / "suite.json").is_file()
    assert "| exact |" in (output / "report.md").read_text(encoding="utf-8")


def test_benchmark_reference_verifier_checks_correctness_and_reports_runtime() -> None:
    suite = {
        "schema_version": "clingrounder.benchmark-suite.v1",
        "benchmark": {"id": "fixture"},
        "split": "test",
        "git_commit": "candidate-commit",
        "runs": {
            "exact": {
                "metrics": {
                    "entity_exact_micro_f1": 0.8,
                    "relation_micro_f1": 1.0,
                },
                "performance": {"document_latency_ms": {"p95": 42.0}},
            }
        },
    }
    reference = {
        "benchmark": "fixture",
        "measurement": {"commit": "reference-commit"},
        "results": [
            {
                "variant": "exact",
                "split": "test",
                "entity_exact_micro_f1": 0.8,
                "relation_micro_f1": 1.0,
                "p95_ms": 10.0,
            }
        ],
    }

    report = verify_dataset_benchmark_reference(suite, reference)

    assert report["verified"] is True
    assert report["runtime_checked"] is False
    assert report["reference_commit"] == "reference-commit"
    assert report["suite_commit"] == "candidate-commit"
    assert report["variants"]["exact"]["runtime"]["measured_p95_ms"] == 42.0


def test_benchmark_reference_verifier_rejects_correctness_drift() -> None:
    suite = {
        "benchmark": {"id": "fixture"},
        "split": "test",
        "runs": {
            "exact": {
                "metrics": {"entity_exact_micro_f1": 0.7},
                "performance": {"document_latency_ms": {"p95": 42.0}},
            }
        },
    }
    reference = {
        "benchmark": "fixture",
        "results": [
            {"variant": "exact", "split": "test", "entity_exact_micro_f1": 0.8}
        ],
    }

    report = verify_dataset_benchmark_reference(suite, reference)

    assert report["verified"] is False
    assert report["variants"]["exact"]["checks"]["entity_exact_micro_f1"] is False


def test_benchmark_reference_verifier_checks_snapshot_and_runtime_provenance() -> None:
    suite = {
        "benchmark": {"id": "fixture", "version": "2.0.0"},
        "split": "test",
        "benchmark_manifest_sha256": "manifest-sha",
        "input_sha256": "input-sha",
        "runs": {
            "exact": {
                "config_fingerprint": "config-sha",
                "terminology_fingerprint": "terminology-sha",
                "metrics": {"entity_exact_micro_f1": 0.8},
                "performance": {"document_latency_ms": {"p95": 42.0}},
            }
        },
    }
    reference = {
        "benchmark": "fixture",
        "dataset": {
            "version": "2.0.0",
            "benchmark_manifest_sha256": "manifest-sha",
            "input_sha256": "different-input",
        },
        "results": [
            {
                "variant": "exact",
                "split": "test",
                "config_fingerprint": "config-sha",
                "terminology_fingerprint": "terminology-sha",
                "entity_exact_micro_f1": 0.8,
            }
        ],
    }

    report = verify_dataset_benchmark_reference(suite, reference)

    assert report["verified"] is False
    assert report["dataset_checks"] == {
        "version": True,
        "benchmark_manifest_sha256": True,
        "input_sha256": False,
    }
    assert report["variants"]["exact"]["provenance_checks"] == {
        "config_fingerprint": True,
        "terminology_fingerprint": True,
    }


def test_benchmark_reference_verifier_rejects_duplicate_variants() -> None:
    suite = {
        "benchmark": {"id": "fixture"},
        "split": "test",
        "runs": {},
    }
    reference = {
        "benchmark": "fixture",
        "results": [
            {"variant": "exact", "split": "test"},
            {"variant": "exact", "split": "test"},
        ],
    }

    with pytest.raises(ValueError, match="duplicate variant"):
        verify_dataset_benchmark_reference(suite, reference)


def test_benchmark_suite_rejects_path_traversal_names(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="Invalid benchmark suite config name"):
        run_dataset_benchmark_suite(
            "benchmarks/vi_clinical_grounding_v1",
            {"../escape": "configs/benchmarks/vi_clinical_grounding_v1/exact.yaml"},
            tmp_path / "suite",
        )


def test_benchmark_taxonomy_comes_from_manifest(tmp_path: Path) -> None:
    benchmark = tmp_path / "finding-task"
    benchmark.mkdir()
    row = {
        "document_id": "finding-1",
        "text": "Lao phổi.",
        "metadata": {"template_group": "finding"},
        "entities": [
            {
                "id": "e1",
                "span": [0, 8],
                "text": "Lao phổi",
                "type": "FINDING",
                "assertion": "PRESENT",
                "code_system": "NONE",
                "code": None,
            }
        ],
        "relations": [],
    }
    (benchmark / "test.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (benchmark / "dataset_manifest.yaml").write_text(
        """
schema_version: clingrounder.dataset-manifest.v1
dataset:
  id: finding-task
  version: '1.0.0'
  status: synthetic_pilot
  license: MIT
  license_url: https://opensource.org/license/mit
  human_reviewed: false
splits:
  test:
    path: test.jsonl
    documents: 1
    sha256: placeholder
entities: [FINDING]
assertions: [PRESENT]
code_systems: [NONE]
policy:
  template_grouping_required: true
  test_used_for_development: false
  private_data: false
""",
        encoding="utf-8",
    )
    import hashlib

    manifest_path = benchmark / "dataset_manifest.yaml"
    manifest = manifest_path.read_text(encoding="utf-8").replace(
        "sha256: placeholder",
        f"sha256: {hashlib.sha256((benchmark / 'test.jsonl').read_bytes()).hexdigest()}",
    )
    manifest_path.write_text(manifest, encoding="utf-8")

    summary = run_dataset_benchmark(
        benchmark,
        "configs/benchmarks/vi_clinical_grounding_v1/exact.yaml",
        tmp_path / "run",
    )

    assert summary["metrics"]["entity_by_type"]["FINDING"]["gold"] == 1
