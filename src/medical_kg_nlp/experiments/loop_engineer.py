from __future__ import annotations

from datetime import date
from typing import Any

from medical_kg_nlp.experiments.loop_analysis import (
    baseline_report_id,
    context_confusion_rows,
    decide_experiment,
    metric_delta,
    metric_snapshot,
    prioritize_errors,
    recommend_next_experiment,
    top_error_cases,
)
from medical_kg_nlp.experiments.loop_agent import (
    build_agent_actions,
    build_agent_context,
    build_agent_poll_state,
    render_agent_brief,
    render_agent_compact_markdown,
)
from medical_kg_nlp.experiments.loop_artifacts import (
    render_decision_markdown,
    render_next_experiment_markdown,
    render_top_error_cases_markdown,
    write_loop_engineering_report,
)

__all__ = [
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
]


def build_loop_engineering_report(
    current_report: dict[str, Any],
    *,
    baseline_report: dict[str, Any] | None = None,
    experiment_id: str,
    module: str,
    hypothesis: str,
    changes: list[str],
    owner: str = "",
    dataset: dict[str, str] | None = None,
    notes: list[str] | None = None,
    primary_metric: str = "loop_score",
    keep_delta: float = 0.001,
    revert_delta: float = 0.001,
    today: date | None = None,
    top_k: int = 30,
) -> dict[str, Any]:
    current_metrics = metric_snapshot(current_report)
    baseline_metrics = metric_snapshot(baseline_report) if baseline_report else {}
    delta = metric_delta(current_metrics, baseline_metrics)
    top_errors = prioritize_errors(current_report, top_k=top_k)
    top_cases = top_error_cases(current_report, top_errors, top_k=top_k)
    next_experiment = recommend_next_experiment(top_errors, current_report)
    decision = decide_experiment(
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics,
        primary_metric=primary_metric,
        keep_delta=keep_delta,
        revert_delta=revert_delta,
    )
    log = {
        "id": experiment_id,
        "date": (today or date.today()).isoformat(),
        "owner": owner,
        "module": module,
        "hypothesis": hypothesis,
        "change": changes,
        "baseline": baseline_report_id(baseline_report),
        "dataset": dataset or {},
        "metrics_before": baseline_metrics,
        "metrics_after": current_metrics,
        "metric_delta": delta,
        "decision": decision,
        "notes": notes or [],
        "next": next_experiment,
    }
    decision_payload = {
        "decision": decision,
        "primary_metric": primary_metric,
        "primary_before": baseline_metrics.get(primary_metric),
        "primary_after": current_metrics.get(primary_metric),
        "primary_delta": delta.get(primary_metric),
        "blocking_issues": current_metrics.get("validation_issue_count", 0.0),
    }
    agent_actions = build_agent_actions(
        top_errors=top_errors,
        next_experiment=next_experiment,
        decision=decision_payload,
        current_metrics=current_metrics,
        top_error_cases_payload=top_cases,
    )
    agent_context = build_agent_context(
        experiment_log=log,
        decision=decision_payload,
        current_metrics=current_metrics,
        top_errors=top_errors,
        next_experiment=next_experiment,
    )
    agent_poll = build_agent_poll_state(agent_context, agent_actions)
    return {
        "experiment_log": log,
        "decision": decision_payload,
        "top_errors": top_errors,
        "next_experiment": next_experiment,
        "context_confusion_matrix": context_confusion_rows(current_report),
        "top_error_cases": top_cases,
        "agent": {
            "context": agent_context,
            "actions": agent_actions,
            "poll": agent_poll,
            "brief": render_agent_brief(agent_context, agent_actions),
            "compact": render_agent_compact_markdown(agent_context, agent_actions, agent_poll),
        },
    }
