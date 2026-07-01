from __future__ import annotations

from typing import Any

from medical_kg_nlp.evaluation.loop_policy import (
    CORE_SCORE_METRICS,
    DEFAULT_ERROR_POLICY,
    ERROR_POLICIES,
)


def metric_snapshot(report: dict[str, Any] | None) -> dict[str, float]:
    if report is None:
        return {}
    metrics = _mapping(report.get("metrics", {}))
    summary = _mapping(report.get("summary", {}))
    validation = _mapping(_mapping(report.get("validation", {})).get("summary", {}))
    error_summary = _mapping(report.get("error_summary", {}))
    candidate_metrics = _mapping(report.get("candidate_metrics", {}))
    snapshot: dict[str, float] = {
        "span_exact_f1": _number_at(metrics, ["span_exact", "f1"]),
        "span_exact_precision": _number_at(metrics, ["span_exact", "precision"]),
        "span_exact_recall": _number_at(metrics, ["span_exact", "recall"]),
        "span_overlap_f1": _number_at(metrics, ["span_overlap", "f1"]),
        "linking_accuracy_at_1": _number_at(metrics, ["linking_accuracy_at_1"]),
        "linking_recall_at_5": _number_at(metrics, ["linking_recall_at_5"]),
        "linking_recall_at_10": _number_at(metrics, ["linking_recall_at_10"]),
        "linking_recall_at_20": _number_at(metrics, ["linking_recall_at_20"]),
        "linking_mrr": _number_at(metrics, ["linking_mrr"]),
        "context_accuracy": _number_at(metrics, ["context_accuracy"]),
        "context_macro_f1": _number_at(metrics, ["context_macro_f1"]),
        "relation_f1": _number_at(metrics, ["relation", "f1"]),
        "validation_issue_count": _number_at(validation, ["issue_count"]),
        "error_count": _number_at(summary, ["error_count"]),
        "candidate_missing_gold": _number_at(error_summary, ["candidate_missing_gold"]),
        "candidate_empty": _number_at(error_summary, ["candidate_empty"]),
        "linking_wrong_top1": _number_at(error_summary, ["linking_wrong_top1"]),
        "severe_context_error": _number_at(error_summary, ["severe_context_error"]),
        "span_boundary": _number_at(error_summary, ["span_boundary"]),
        "missing_entity": _number_at(error_summary, ["missing_entity"]),
        "spurious_entity": _number_at(error_summary, ["spurious_entity"]),
        "invalid_relation": _number_at(error_summary, ["invalid_relation"]),
        "entities_with_no_candidates": _number_at(candidate_metrics, ["entities_with_no_candidates"]),
    }
    core_values = [snapshot[key] for key in CORE_SCORE_METRICS]
    snapshot["loop_score"] = round(sum(core_values) / len(core_values), 6)
    return snapshot


def metric_delta(current_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(current_metrics) | set(baseline_metrics))
    return {key: round(current_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0), 6) for key in keys}


def decide_experiment(
    *,
    current_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    primary_metric: str,
    keep_delta: float,
    revert_delta: float,
) -> str:
    current_validation = current_metrics.get("validation_issue_count", 0.0)
    baseline_validation = baseline_metrics.get("validation_issue_count", 0.0)
    if not baseline_metrics:
        return "baseline" if current_validation == 0 else "refine"
    if current_validation > baseline_validation:
        return "revert"
    if current_validation > 0:
        return "refine"
    primary_delta = current_metrics.get(primary_metric, 0.0) - baseline_metrics.get(primary_metric, 0.0)
    if primary_delta >= keep_delta:
        return "keep"
    if primary_delta <= -revert_delta:
        return "revert"
    return "refine"


def prioritize_errors(report: dict[str, Any], *, top_k: int = 30) -> list[dict[str, Any]]:
    error_summary = _mapping(report.get("error_summary", {}))
    total = sum(_int_value(value) for value in error_summary.values())
    rows: list[dict[str, Any]] = []
    if total == 0:
        return rows
    for error_type, value in error_summary.items():
        count = _int_value(value)
        policy = ERROR_POLICIES.get(str(error_type), DEFAULT_ERROR_POLICY)
        frequency = count / total
        rows.append(
            {
                "error_type": str(error_type),
                "count": count,
                "frequency": round(frequency, 6),
                "priority": round(policy.impact * frequency * policy.fixability / policy.cost, 6),
                "module": policy.module,
                "impact": policy.impact,
                "fixability": policy.fixability,
                "cost": policy.cost,
                "recommendation": policy.recommendation,
                "success_metric": policy.success_metric,
            }
        )
    return sorted(rows, key=lambda row: (-float(row["priority"]), str(row["error_type"])))[:top_k]


def recommend_next_experiment(
    top_errors: list[dict[str, Any]],
    current_report: dict[str, Any],
) -> dict[str, Any]:
    if not top_errors:
        return {
            "module": "evaluation",
            "target_error": None,
            "hypothesis": "Current run has no structured errors; compare on a harder validation split.",
            "change": ["Run the same pipeline on local_holdout or add harder regression cases."],
            "success_metric": "loop_score",
            "success_criteria": "Hold validation issues at 0 while preserving or improving loop_score.",
        }
    top = top_errors[0]
    success_metric = str(top["success_metric"])
    error_type = str(top["error_type"])
    return {
        "module": str(top["module"]),
        "target_error": error_type,
        "hypothesis": f"Reducing {error_type} will improve {success_metric} without increasing validation issues.",
        "change": [
            str(top["recommendation"]),
            "Inspect top_error_cases.md and add one focused regression test before changing code.",
        ],
        "success_metric": success_metric,
        "success_criteria": _success_criteria(success_metric, current_report),
    }


def context_confusion_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = _mapping(_mapping(report.get("metrics", {})).get("context_confusion_matrix", {}))
    rows: list[dict[str, Any]] = []
    for gold_label, predictions in sorted(matrix.items()):
        prediction_counts = _mapping(predictions)
        for pred_label, count in sorted(prediction_counts.items()):
            rows.append({"gold": str(gold_label), "prediction": str(pred_label), "count": _int_value(count)})
    return rows


def top_error_cases(
    report: dict[str, Any],
    top_errors: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    errors = _dict_list(report.get("errors", []))
    selected_types = {str(row["error_type"]) for row in top_errors[:5]}
    cases: list[dict[str, Any]] = []
    for row in errors:
        if selected_types and str(row.get("error_type")) not in selected_types:
            continue
        cases.append(row)
        if len(cases) >= top_k:
            break
    return cases


def baseline_report_id(baseline_report: dict[str, Any] | None) -> str | None:
    if baseline_report is None:
        return None
    summary = _mapping(baseline_report.get("summary", {}))
    return str(summary.get("run_id") or summary.get("pipeline_version") or "baseline")


def _success_criteria(success_metric: str, current_report: dict[str, Any]) -> str:
    metrics = metric_snapshot(current_report)
    current_value = metrics.get(success_metric)
    if success_metric == "validation_issue_count":
        return "Validation issue count must reach 0."
    if current_value is None:
        return f"{success_metric} should improve without increasing validation issues."
    return f"{success_metric} should exceed current value {_format_value(current_value)}."


def _number_at(payload: dict[str, Any], path: list[str]) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _format_value(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.6f}"
    if value is None:
        return "N/A"
    return str(value)
