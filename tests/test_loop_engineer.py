from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from medical_kg_nlp.experiments import loop_analysis, loop_artifacts
from medical_kg_nlp.experiments import loop_engineer as loop_engineer_module
from medical_kg_nlp.experiments.loop_engineer import (
    build_loop_engineering_report,
    metric_snapshot,
    write_loop_engineering_report,
)
from medical_kg_nlp.experiments.loop_policy import AGENT_PLAYBOOKS


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
    assert loop_report["agent"]["actions"][0]["recommended_files"]
    assert "Validation issue count must not increase." in loop_report["agent"]["actions"][0]["acceptance_criteria"]
    assert "uv run pytest tests/test_candidate_generation.py tests/test_dictionary.py -q" in (
        loop_report["agent"]["actions"][0]["commands"]
    )
    assert loop_report["agent"]["poll"]["status"] == "ready_for_agent"
    assert loop_report["agent"]["poll"]["read_order"][:2] == ["agent_poll.json", "agent_compact.md"]
    assert "Read agent_compact.md first" in loop_report["agent"]["poll"]["token_strategy"]

    journal_dir = tmp_path / "journal"
    write_loop_engineering_report(loop_report, tmp_path, journal_dir=journal_dir)
    for filename in [
        "loop_report.json",
        "experiment_log.yaml",
        "experiment_log.json",
        "agent_poll.json",
        "agent_actions.jsonl",
        "agent_brief.md",
        "agent_compact.md",
        "confusion_matrix.csv",
        "decision.md",
        "next_experiment.md",
        "top_error_cases.md",
    ]:
        assert (tmp_path / filename).exists()
    for filename in [
        "experiments.jsonl",
        "experiment_index.json",
        "experiment_memory.json",
        "experiment_notebook.md",
    ]:
        assert (journal_dir / filename).exists()
    memory = json.loads((journal_dir / "experiment_memory.json").read_text(encoding="utf-8"))
    assert memory["reuse"][0]["id"] == "N002"
    poll = json.loads((tmp_path / "agent_poll.json").read_text(encoding="utf-8"))
    assert poll["artifact_paths"]["journal_memory"] == str(journal_dir / "experiment_memory.json")


def test_loop_engineer_reexports_split_modules() -> None:
    assert loop_engineer_module.metric_snapshot is loop_analysis.metric_snapshot
    assert loop_engineer_module.prioritize_errors is loop_analysis.prioritize_errors
    assert loop_engineer_module.write_loop_engineering_report is loop_artifacts.write_loop_engineering_report
    assert loop_engineer_module.render_decision_markdown is loop_artifacts.render_decision_markdown
    assert set(loop_engineer_module.__all__) == {
        "baseline_report_id",
        "build_loop_engineering_report",
        "context_confusion_rows",
        "decide_experiment",
        "metric_delta",
        "metric_snapshot",
        "prioritize_errors",
        "recommend_next_experiment",
        "render_decision_markdown",
        "render_next_experiment_markdown",
        "render_top_error_cases_markdown",
        "top_error_cases",
        "write_loop_engineering_report",
    }
    assert set(AGENT_PLAYBOOKS["evaluation"].focus_files) >= {
        "src/medical_kg_nlp/experiments/loop_engineer.py",
        "src/medical_kg_nlp/experiments/loop_analysis.py",
        "src/medical_kg_nlp/experiments/loop_artifacts.py",
        "src/medical_kg_nlp/experiments/loop_agent.py",
        "src/medical_kg_nlp/experiments/loop_journal.py",
        "src/medical_kg_nlp/experiments/loop_policy.py",
    }


def test_loop_engineer_evaluation_path_uses_split_module_playbook() -> None:
    loop_report = build_loop_engineering_report(
        _report(loop_score_shift=0.0),
        experiment_id="E001",
        module="evaluation",
        hypothesis="Baseline has no structured errors.",
        changes=["Generate loop report for a clean run."],
    )

    action = loop_report["agent"]["actions"][0]
    assert loop_report["next_experiment"]["module"] == "evaluation"
    assert loop_report["top_error_cases"] == []
    assert set(action["recommended_files"]) >= {
        "src/medical_kg_nlp/experiments/loop_engineer.py",
        "src/medical_kg_nlp/experiments/loop_analysis.py",
        "src/medical_kg_nlp/experiments/loop_artifacts.py",
        "src/medical_kg_nlp/experiments/loop_agent.py",
        "src/medical_kg_nlp/experiments/loop_journal.py",
        "src/medical_kg_nlp/experiments/loop_policy.py",
    }
    assert "src/medical_kg_nlp/experiments/loop_analysis.py" in loop_report["agent"]["brief"]
    assert "uv run pytest tests/test_phase1.py tests/test_pipeline_report.py tests/test_loop_engineer.py -q" in (
        action["commands"]
    )


def test_loop_engineer_prefers_phase1_score_when_report_has_phase1_metrics() -> None:
    report = _report(loop_score_shift=0.0)
    report["metrics"]["relation"]["f1"] = 0.0
    report["phase1"] = {
        "metrics": {
            "score": 88.0,
            "text_score": 0.9,
            "assertions_score": 0.8,
            "candidates_score": 0.92,
        },
        "validation_summary": {"issue_count": 2, "by_kind": {"phase1_offset": 2}},
    }

    snapshot = metric_snapshot(report)

    assert snapshot["loop_score"] == 0.88
    assert snapshot["phase1_score"] == 88.0
    assert snapshot["validation_issue_count"] == 2


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
    assert loop_report["agent"]["poll"]["status"] == "blocked_by_validation"


@pytest.mark.integration
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
    journal_dir = tmp_path / "journal"
    assert loop_report["decision"]["decision"] == "keep"
    assert "severe_context_error" in (output_dir / "top_error_cases.md").read_text(encoding="utf-8")
    assert "Acceptance criteria" in (output_dir / "agent_brief.md").read_text(encoding="utf-8")
    assert "Read Order" in (output_dir / "agent_compact.md").read_text(encoding="utf-8")
    assert json.loads((output_dir / "agent_poll.json").read_text(encoding="utf-8"))[
        "poll_interval_seconds"
    ] == 30
    assert json.loads((output_dir / "agent_actions.jsonl").read_text(encoding="utf-8").splitlines()[0])[
        "module"
    ] == "context"
    assert (journal_dir / "experiments.jsonl").exists()
    notebook = (journal_dir / "experiment_notebook.md").read_text(encoding="utf-8")
    assert "C001 - keep" in notebook


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
