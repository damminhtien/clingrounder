from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from medical_kg_nlp.evaluation.loop_engineer import (
    build_loop_engineering_report,
    metric_snapshot,
    write_loop_engineering_report,
)


def test_loop_engineer_keeps_improved_valid_experiment(tmp_path: Path) -> None:
    baseline = _report(loop_score_shift=0.0, errors={"candidate_missing_gold": 4})
    current = _report(loop_score_shift=0.02, errors={"candidate_missing_gold": 2})

    loop_report = build_loop_engineering_report(
        current,
        baseline_report=baseline,
        experiment_id="N002",
        module="normalization",
        hypothesis="Adding Vietnamese aliases improves candidate recall.",
        changes=["Add two Vietnamese disease aliases."],
        owner="tester",
        dataset={"valid": "sample"},
    )

    assert loop_report["decision"]["decision"] == "keep"
    assert loop_report["experiment_log"]["metric_delta"]["loop_score"] == 0.02
    assert loop_report["top_errors"][0]["error_type"] == "candidate_missing_gold"
    assert loop_report["next_experiment"]["module"] == "normalization"

    write_loop_engineering_report(loop_report, tmp_path)
    for filename in [
        "loop_report.json",
        "experiment_log.yaml",
        "experiment_log.json",
        "confusion_matrix.csv",
        "decision.md",
        "next_experiment.md",
        "top_error_cases.md",
    ]:
        assert (tmp_path / filename).exists()


def test_loop_engineer_reverts_when_validation_gets_worse() -> None:
    baseline = _report(loop_score_shift=0.0, validation_issues=0)
    current = _report(loop_score_shift=0.05, validation_issues=1, errors={"invalid_code_system": 1})

    loop_report = build_loop_engineering_report(
        current,
        baseline_report=baseline,
        experiment_id="K001",
        module="kg_validation",
        hypothesis="Looser KG constraints raise recall.",
        changes=["Disable entity code-system validation."],
    )

    assert loop_report["decision"]["decision"] == "revert"
    assert metric_snapshot(current)["validation_issue_count"] == 1
    assert loop_report["top_errors"][0]["module"] == "kg_validation"


def test_loop_engineer_cli_writes_decision_artifacts(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    output_dir = tmp_path / "loop"
    baseline_path.write_text(json.dumps(_report(loop_score_shift=0.0)), encoding="utf-8")
    current_path.write_text(
        json.dumps(_report(loop_score_shift=0.01, errors={"severe_context_error": 3})),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/loop_engineer.py",
            "--current-report",
            str(current_path),
            "--baseline-report",
            str(baseline_path),
            "--output-dir",
            str(output_dir),
            "--experiment-id",
            "C001",
            "--module",
            "context",
            "--hypothesis",
            "Negation rule improves severe context errors.",
            "--change",
            "Add clause-scoped negation rule.",
            "--dataset",
            "valid=sample",
        ],
        check=True,
    )

    loop_report = json.loads((output_dir / "loop_report.json").read_text(encoding="utf-8"))
    assert loop_report["decision"]["decision"] == "keep"
    assert "severe_context_error" in (output_dir / "top_error_cases.md").read_text(encoding="utf-8")


def _report(
    *,
    loop_score_shift: float,
    validation_issues: int = 0,
    errors: dict[str, int] | None = None,
) -> dict[str, object]:
    span = 0.7 + loop_score_shift
    linking = 0.6 + loop_score_shift
    context = 0.8 + loop_score_shift
    relation = 0.5 + loop_score_shift
    error_summary = errors or {}
    return {
        "summary": {
            "document_count": 2,
            "prediction_count": 2,
            "error_count": sum(error_summary.values()),
        },
        "metrics": {
            "span_exact": {"precision": span, "recall": span, "f1": span},
            "span_overlap": {"precision": span, "recall": span, "f1": span},
            "linking_accuracy_at_1": linking,
            "linking_recall_at_5": linking,
            "linking_recall_at_10": linking,
            "linking_recall_at_20": linking,
            "linking_mrr": linking,
            "context_accuracy": context,
            "context_macro_f1": context,
            "context_confusion_matrix": {"NEGATED": {"PRESENT": error_summary.get("severe_context_error", 0)}},
            "relation": {"precision": relation, "recall": relation, "f1": relation},
        },
        "validation": {
            "summary": {
                "issue_count": validation_issues,
                "by_kind": {"invalid_code_system": validation_issues} if validation_issues else {},
            },
            "issues": [],
        },
        "error_summary": error_summary,
        "candidate_metrics": {"entities_with_no_candidates": 0},
        "errors": [
            {
                "document_id": "sample",
                "stage": "context_assertion_classification",
                "error_type": error_type,
                "severity": "error",
                "span": [0, 5],
                "text_window": "Không ghi nhận hen phế quản.",
                "gold": {},
                "prediction": {},
                "candidate_rank": None,
                "candidate_list": [],
                "validation_path": "",
                "notes": "Synthetic loop test case.",
            }
            for error_type, count in error_summary.items()
            for _ in range(count)
        ],
    }
