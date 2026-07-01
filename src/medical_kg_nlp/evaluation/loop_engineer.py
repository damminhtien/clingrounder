from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


CORE_SCORE_METRICS = (
    "span_exact_f1",
    "linking_accuracy_at_1",
    "context_macro_f1",
    "relation_f1",
)


@dataclass(frozen=True)
class ErrorPolicy:
    module: str
    impact: float
    fixability: float
    cost: float
    recommendation: str
    success_metric: str


@dataclass(frozen=True)
class AgentPlaybook:
    focus_files: tuple[str, ...]
    commands: tuple[str, ...]
    guardrails: tuple[str, ...]


AGENT_PLAYBOOKS: dict[str, AgentPlaybook] = {
    "schema": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/schema/validator.py",
            "src/medical_kg_nlp/schema/annotation.py",
            "tests/test_prediction_validator.py",
        ),
        commands=(
            "uv run pytest tests/test_prediction_validator.py -q",
            "uv run mypy src",
        ),
        guardrails=(
            "Keep exported prediction fields backward-compatible.",
            "Schema or enum changes require focused tests and docs.",
        ),
    ),
    "preprocessing": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/preprocessing/offset_mapping.py",
            "src/medical_kg_nlp/preprocessing/sentence_splitter.py",
            "tests/test_offset_mapping.py",
        ),
        commands=(
            "uv run pytest tests/test_offset_mapping.py -q",
            "uv run pytest tests/test_pipeline_tracing.py -q",
        ),
        guardrails=(
            "Never destroy or rewrite original character offsets.",
            "Normalized text is lookup-only unless an explicit offset map is used.",
        ),
    ),
    "entity_extraction": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/ner/rule_ner.py",
            "src/medical_kg_nlp/dictionaries/dictionary_store.py",
            "tests/test_pipeline_smoke.py",
            "tests/test_dictionary.py",
        ),
        commands=(
            "uv run pytest tests/test_pipeline_smoke.py tests/test_dictionary.py -q",
            "uv run pytest tests/test_offset_mapping.py -q",
        ),
        guardrails=(
            "Every emitted span must validate against the original source text.",
            "Avoid broad NER refactors unless the top error cases prove they are needed.",
        ),
    ),
    "candidate_generation": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/retrieval/candidate_generator.py",
            "src/medical_kg_nlp/dictionaries/dictionary_store.py",
            "src/medical_kg_nlp/linking/linker.py",
            "tests/test_candidate_generation.py",
        ),
        commands=(
            "uv run pytest tests/test_candidate_generation.py tests/test_dictionary.py -q",
            "uv run pytest tests/test_prediction_validator.py -q",
        ),
        guardrails=(
            "Candidate generation must filter by entity type before final linking.",
            "Never emit candidate codes outside the loaded dictionary.",
        ),
    ),
    "normalization": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/linking/linker.py",
            "src/medical_kg_nlp/retrieval/candidate_generator.py",
            "src/medical_kg_nlp/dictionaries/dictionary_store.py",
            "tests/test_candidate_generation.py",
        ),
        commands=(
            "uv run pytest tests/test_candidate_generation.py tests/test_dictionary.py -q",
            "uv run pytest tests/test_prediction_validator.py -q",
        ),
        guardrails=(
            "Never map DRUG entities to ICD-10 or DISEASE entities to RxNorm.",
            "Candidate recall@20 must be fixed before reranker tuning can help.",
        ),
    ),
    "context": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/context/rules.py",
            "src/medical_kg_nlp/context/assertion.py",
            "tests/test_context_rules.py",
            "tests/test_context_metrics.py",
        ),
        commands=(
            "uv run pytest tests/test_context_rules.py tests/test_context_metrics.py -q",
            "uv run pytest tests/test_pipeline_smoke.py -q",
        ),
        guardrails=(
            "Negated diseases must not become confirmed patient conditions.",
            "Family-history diseases must not become patient-present diseases.",
        ),
    ),
    "relation_extraction": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/relations/rule_relations.py",
            "src/medical_kg_nlp/relations/candidate_pairs.py",
            "src/medical_kg_nlp/kg/constraints.py",
            "tests/test_kg_constraints.py",
        ),
        commands=(
            "uv run pytest tests/test_kg_constraints.py -q",
            "uv run pytest tests/test_pipeline_smoke.py -q",
        ),
        guardrails=(
            "Relation endpoint types must satisfy KG constraints.",
            "Do not keep impossible relations for score-chasing.",
        ),
    ),
    "kg_validation": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/kg/constraints.py",
            "src/medical_kg_nlp/kg/validator.py",
            "tests/test_kg_constraints.py",
            "tests/test_prediction_validator.py",
        ),
        commands=(
            "uv run pytest tests/test_kg_constraints.py tests/test_prediction_validator.py -q",
            "uv run mypy src",
        ),
        guardrails=(
            "Ontology/KG violations are blocking, not cosmetic.",
            "Invalid code systems should be rejected or reset before export.",
        ),
    ),
    "evaluation": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/evaluation/pipeline_report.py",
            "src/medical_kg_nlp/evaluation/loop_engineer.py",
            "tests/test_pipeline_report.py",
            "tests/test_loop_engineer.py",
        ),
        commands=(
            "uv run pytest tests/test_pipeline_report.py tests/test_loop_engineer.py -q",
            "uv run ruff check .",
        ),
        guardrails=(
            "Do not hide validation failures behind aggregate metrics.",
            "Keep reports machine-readable and stable for agents.",
        ),
    ),
}


DEFAULT_AGENT_PLAYBOOK = AgentPlaybook(
    focus_files=("docs/evaluation.md", "src/medical_kg_nlp/evaluation/pipeline_report.py"),
    commands=("uv run pytest tests -q",),
    guardrails=(
        "Make one meaningful change per experiment.",
        "Add focused tests before relying on a metric improvement.",
    ),
)


ERROR_POLICIES: dict[str, ErrorPolicy] = {
    "schema": ErrorPolicy(
        module="schema",
        impact=1.0,
        fixability=0.95,
        cost=1.0,
        recommendation="Fix schema parsing/export before comparing model metrics.",
        success_metric="validation_issue_count",
    ),
    "offset": ErrorPolicy(
        module="preprocessing",
        impact=1.0,
        fixability=0.9,
        cost=1.0,
        recommendation="Fix offset preservation or span trimming regression.",
        success_metric="validation_issue_count",
    ),
    "invalid_code_system": ErrorPolicy(
        module="kg_validation",
        impact=1.0,
        fixability=0.9,
        cost=1.0,
        recommendation="Tighten entity type to code-system constraints.",
        success_metric="validation_issue_count",
    ),
    "unknown_dictionary_code": ErrorPolicy(
        module="normalization",
        impact=1.0,
        fixability=0.85,
        cost=1.0,
        recommendation="Force linked codes and candidates to come from the loaded dictionary.",
        success_metric="validation_issue_count",
    ),
    "invalid_candidate_code_system": ErrorPolicy(
        module="candidate_generation",
        impact=0.95,
        fixability=0.9,
        cost=1.0,
        recommendation="Filter candidates by entity type before ranking.",
        success_metric="candidate_missing_gold",
    ),
    "candidate_missing_gold": ErrorPolicy(
        module="normalization",
        impact=0.95,
        fixability=0.75,
        cost=1.4,
        recommendation="Improve dictionary aliases or retrieval sources so gold codes enter top-k.",
        success_metric="linking_recall_at_20",
    ),
    "candidate_empty": ErrorPolicy(
        module="normalization",
        impact=0.95,
        fixability=0.8,
        cost=1.2,
        recommendation="Add exact, abbreviation, fuzzy, or n-gram coverage for empty candidate lists.",
        success_metric="linking_recall_at_20",
    ),
    "linking_wrong_top1": ErrorPolicy(
        module="normalization",
        impact=0.9,
        fixability=0.65,
        cost=1.6,
        recommendation="Tune reranking, context features, or score blending after candidate recall is healthy.",
        success_metric="linking_accuracy_at_1",
    ),
    "linking_unlinked": ErrorPolicy(
        module="normalization",
        impact=0.9,
        fixability=0.75,
        cost=1.2,
        recommendation="Inspect assignment thresholds and dictionary coverage for unlinked gold entities.",
        success_metric="linking_accuracy_at_1",
    ),
    "severe_context_error": ErrorPolicy(
        module="context",
        impact=0.95,
        fixability=0.8,
        cost=1.2,
        recommendation="Prioritize negation, family-history, and uncertainty rules before larger models.",
        success_metric="context_macro_f1",
    ),
    "context_confusion": ErrorPolicy(
        module="context",
        impact=0.8,
        fixability=0.75,
        cost=1.3,
        recommendation="Add cue-specific regression cases and section/sentence scoped rules.",
        success_metric="context_macro_f1",
    ),
    "span_boundary": ErrorPolicy(
        module="entity_extraction",
        impact=0.9,
        fixability=0.65,
        cost=1.4,
        recommendation="Tune tokenizer, dictionary span selection, or postprocess merge/split rules.",
        success_metric="span_exact_f1",
    ),
    "missing_entity": ErrorPolicy(
        module="entity_extraction",
        impact=0.9,
        fixability=0.7,
        cost=1.3,
        recommendation="Increase span recall with aliases, abbreviations, or recall-oriented NER rules.",
        success_metric="span_exact.recall",
    ),
    "spurious_entity": ErrorPolicy(
        module="entity_extraction",
        impact=0.8,
        fixability=0.75,
        cost=1.2,
        recommendation="Add blocklist, section prior, or confidence threshold for false positive mentions.",
        success_metric="span_exact.precision",
    ),
    "type_confusion": ErrorPolicy(
        module="entity_extraction",
        impact=0.85,
        fixability=0.75,
        cost=1.2,
        recommendation="Add type-specific aliases or postprocess type disambiguation.",
        success_metric="span_exact_f1",
    ),
    "invalid_relation": ErrorPolicy(
        module="kg_validation",
        impact=0.9,
        fixability=0.9,
        cost=1.0,
        recommendation="Apply relation endpoint/type constraints before export.",
        success_metric="validation_issue_count",
    ),
    "missing_relation": ErrorPolicy(
        module="relation_extraction",
        impact=0.75,
        fixability=0.65,
        cost=1.5,
        recommendation="Increase candidate pair recall or add relation rules for frequent missing types.",
        success_metric="relation_f1",
    ),
    "spurious_relation": ErrorPolicy(
        module="relation_extraction",
        impact=0.75,
        fixability=0.75,
        cost=1.2,
        recommendation="Tighten relation constraints, direction checks, or distance thresholds.",
        success_metric="relation_f1",
    ),
    "relation_type_confusion": ErrorPolicy(
        module="relation_extraction",
        impact=0.75,
        fixability=0.65,
        cost=1.4,
        recommendation="Add type-specific relation features and endpoint constraints.",
        success_metric="relation_f1",
    ),
}


DEFAULT_ERROR_POLICY = ErrorPolicy(
    module="pipeline",
    impact=0.6,
    fixability=0.6,
    cost=1.5,
    recommendation="Inspect examples and add the smallest targeted regression test.",
    success_metric="loop_score",
)


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
        top_error_cases_payload=top_error_cases(current_report, top_errors, top_k=top_k),
    )
    agent_context = build_agent_context(
        experiment_log=log,
        decision=decision_payload,
        current_metrics=current_metrics,
        top_errors=top_errors,
        next_experiment=next_experiment,
    )
    return {
        "experiment_log": log,
        "decision": decision_payload,
        "top_errors": top_errors,
        "next_experiment": next_experiment,
        "context_confusion_matrix": context_confusion_rows(current_report),
        "top_error_cases": top_error_cases(current_report, top_errors, top_k=top_k),
        "agent": {
            "context": agent_context,
            "actions": agent_actions,
            "brief": render_agent_brief(agent_context, agent_actions),
        },
    }


def write_loop_engineering_report(loop_report: dict[str, Any], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "loop_report.json", loop_report)
    _write_yaml(path / "experiment_log.yaml", _mapping(loop_report["experiment_log"]))
    _write_json(path / "experiment_log.json", loop_report["experiment_log"])
    _write_confusion_matrix(path / "confusion_matrix.csv", _list(loop_report["context_confusion_matrix"]))
    _write_jsonl(path / "agent_actions.jsonl", _dict_list(_mapping(loop_report["agent"])["actions"]))
    (path / "decision.md").write_text(render_decision_markdown(loop_report), encoding="utf-8")
    (path / "next_experiment.md").write_text(render_next_experiment_markdown(loop_report), encoding="utf-8")
    (path / "top_error_cases.md").write_text(render_top_error_cases_markdown(loop_report), encoding="utf-8")
    (path / "agent_brief.md").write_text(str(_mapping(loop_report["agent"])["brief"]), encoding="utf-8")


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
    return {
        key: round(current_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0), 6)
        for key in keys
    }


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
        priority = policy.impact * frequency * policy.fixability / policy.cost
        rows.append(
            {
                "error_type": str(error_type),
                "count": count,
                "frequency": round(frequency, 6),
                "priority": round(priority, 6),
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
    module = str(top["module"])
    error_type = str(top["error_type"])
    success_metric = str(top["success_metric"])
    return {
        "module": module,
        "target_error": error_type,
        "hypothesis": (
            f"Reducing {error_type} will improve {success_metric} without increasing "
            "validation issues."
        ),
        "change": [
            str(top["recommendation"]),
            "Inspect top_error_cases.md and add one focused regression test before changing code.",
        ],
        "success_metric": success_metric,
        "success_criteria": _success_criteria(success_metric, current_report),
    }


def build_agent_context(
    *,
    experiment_log: dict[str, Any],
    decision: dict[str, Any],
    current_metrics: dict[str, float],
    top_errors: list[dict[str, Any]],
    next_experiment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "objective": "Run one disciplined experiment loop and make the smallest useful code change.",
        "experiment_id": experiment_log["id"],
        "module": next_experiment["module"],
        "target_error": next_experiment["target_error"],
        "decision": decision["decision"],
        "primary_metric": decision["primary_metric"],
        "blocking_issues": decision["blocking_issues"],
        "current_metrics": current_metrics,
        "top_errors": top_errors[:5],
        "next_experiment": next_experiment,
        "global_guardrails": [
            "Preserve original character offsets.",
            "Never output codes absent from the loaded dictionary.",
            "Never map DRUG to ICD-10 or DISEASE to RxNorm.",
            "Keep negated and family-history diseases distinct from present patient conditions.",
            "Change one meaningful component per experiment.",
        ],
    }


def build_agent_actions(
    *,
    top_errors: list[dict[str, Any]],
    next_experiment: dict[str, Any],
    decision: dict[str, Any],
    current_metrics: dict[str, float],
    top_error_cases_payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    module = str(next_experiment["module"])
    playbook = AGENT_PLAYBOOKS.get(module, DEFAULT_AGENT_PLAYBOOK)
    target_error = next_experiment.get("target_error")
    top_error = top_errors[0] if top_errors else {}
    action = {
        "id": "AGENT-001",
        "status": "ready",
        "priority": 1,
        "module": module,
        "target_error": target_error,
        "title": _agent_action_title(next_experiment),
        "objective": str(next_experiment["hypothesis"]),
        "decision_context": decision,
        "evidence": {
            "top_error": top_error,
            "example_count": len(top_error_cases_payload),
            "sample_cases": _case_summaries(top_error_cases_payload[:5]),
            "current_metric": current_metrics.get(str(next_experiment["success_metric"])),
        },
        "recommended_files": list(playbook.focus_files),
        "implementation_steps": [
            "Read the top error cases before editing code.",
            "Add or update one focused regression test for the target error.",
            "Make the smallest scoped implementation change in the target module.",
            "Run the targeted commands first, then the full verification commands if targeted tests pass.",
            "Regenerate the stage report and loop report to compare against the frozen baseline.",
        ],
        "commands": list(playbook.commands),
        "acceptance_criteria": [
            str(next_experiment["success_criteria"]),
            "Validation issue count must not increase.",
            "Focused tests for the changed behavior must pass.",
            "No offset, dictionary, code-system, context, or KG invariant may regress.",
        ],
        "stop_conditions": [
            "Stop and report if the fix requires a second unrelated component change.",
            "Stop and report if validation issues increase.",
            "Stop and report if evidence points to missing labels or an ambiguous gold annotation.",
            "Stop and report before adding hosted services, Java core, graph databases, or native extensions.",
        ],
        "guardrails": list(playbook.guardrails),
        "required_artifacts_after_change": [
            "updated focused test",
            "stage-wise metrics.json",
            "loop_report.json",
            "decision.md",
        ],
    }
    return [action]


def render_agent_brief(agent_context: dict[str, Any], actions: list[dict[str, Any]]) -> str:
    lines = [
        "# Agent Brief",
        "",
        f"- Objective: {agent_context['objective']}",
        f"- Experiment: {agent_context['experiment_id']}",
        f"- Decision: {agent_context['decision']}",
        f"- Module: {agent_context['module']}",
        f"- Target error: {agent_context['target_error'] or 'N/A'}",
        f"- Primary metric: {agent_context['primary_metric']}",
        f"- Blocking issues: {_format_value(agent_context['blocking_issues'])}",
        "",
        "## Guardrails",
        "",
    ]
    for guardrail in _list(agent_context["global_guardrails"]):
        lines.append(f"- {guardrail}")
    lines.extend(["", "## Actions", ""])
    for action in actions:
        lines.append(f"### {action['id']}: {action['title']}")
        lines.append("")
        lines.append(f"- Status: {action['status']}")
        lines.append(f"- Module: {action['module']}")
        lines.append(f"- Target error: {action['target_error'] or 'N/A'}")
        lines.append(f"- Objective: {action['objective']}")
        lines.append("")
        lines.append("Recommended files:")
        for file_path in _list(action["recommended_files"]):
            lines.append(f"- {file_path}")
        lines.append("")
        lines.append("Commands:")
        for command in _list(action["commands"]):
            lines.append(f"- `{command}`")
        lines.append("")
        lines.append("Acceptance criteria:")
        for criterion in _list(action["acceptance_criteria"]):
            lines.append(f"- {criterion}")
        lines.append("")
        lines.append("Stop conditions:")
        for condition in _list(action["stop_conditions"]):
            lines.append(f"- {condition}")
        lines.append("")
    return "\n".join(lines) + "\n"


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


def render_decision_markdown(loop_report: dict[str, Any]) -> str:
    decision = _mapping(loop_report["decision"])
    experiment = _mapping(loop_report["experiment_log"])
    lines = [
        "# Loop Decision",
        "",
        f"- Experiment: {experiment['id']}",
        f"- Module: {experiment['module']}",
        f"- Decision: {decision['decision']}",
        f"- Primary metric: {decision['primary_metric']}",
        f"- Before: {_format_value(decision['primary_before'])}",
        f"- After: {_format_value(decision['primary_after'])}",
        f"- Delta: {_format_value(decision['primary_delta'])}",
        f"- Blocking issues: {_format_value(decision['blocking_issues'])}",
        "",
        "## Hypothesis",
        "",
        str(experiment["hypothesis"]),
        "",
        "## Change",
        "",
    ]
    for change in _list(experiment["change"]):
        lines.append(f"- {change}")
    return "\n".join(lines) + "\n"


def render_next_experiment_markdown(loop_report: dict[str, Any]) -> str:
    next_experiment = _mapping(loop_report["next_experiment"])
    lines = [
        "# Next Experiment",
        "",
        f"- Module: {next_experiment['module']}",
        f"- Target error: {next_experiment['target_error'] or 'N/A'}",
        f"- Success metric: {next_experiment['success_metric']}",
        f"- Success criteria: {next_experiment['success_criteria']}",
        "",
        "## Hypothesis",
        "",
        str(next_experiment["hypothesis"]),
        "",
        "## Minimal Change",
        "",
    ]
    for change in _list(next_experiment["change"]):
        lines.append(f"- {change}")
    return "\n".join(lines) + "\n"


def render_top_error_cases_markdown(loop_report: dict[str, Any]) -> str:
    top_errors = _dict_list(loop_report["top_errors"])
    cases = _dict_list(loop_report["top_error_cases"])
    lines = ["# Top Error Cases", "", "## Priority", ""]
    if not top_errors:
        lines.append("- No structured errors.")
    else:
        for row in top_errors[:10]:
            lines.append(
                "- "
                f"{row['error_type']}: count={row['count']}, "
                f"frequency={_format_value(row['frequency'])}, "
                f"priority={_format_value(row['priority'])}, module={row['module']}"
            )
    lines.extend(["", "## Cases", ""])
    if not cases:
        lines.append("- No cases selected.")
    else:
        for index, row in enumerate(cases, start=1):
            lines.append(f"### {index}. {row.get('error_type')} [{row.get('document_id')}]")
            lines.append("")
            lines.append(f"- Stage: {row.get('stage')}")
            lines.append(f"- Span: {row.get('span')}")
            lines.append(f"- Notes: {row.get('notes')}")
            window = str(row.get("text_window", "")).replace("\n", " ")
            if window:
                lines.append(f"- Window: {window}")
            lines.append("")
    return "\n".join(lines) + "\n"


def baseline_report_id(baseline_report: dict[str, Any] | None) -> str | None:
    if baseline_report is None:
        return None
    summary = _mapping(baseline_report.get("summary", {}))
    return str(summary.get("run_id") or summary.get("pipeline_version") or "baseline")


def _agent_action_title(next_experiment: dict[str, Any]) -> str:
    target_error = next_experiment.get("target_error")
    if target_error:
        return f"Reduce {target_error}"
    return "Harden evaluation on a more difficult split"


def _case_summaries(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in cases:
        summaries.append(
            {
                "document_id": row.get("document_id"),
                "stage": row.get("stage"),
                "error_type": row.get("error_type"),
                "span": row.get("span"),
                "text_window": str(row.get("text_window", "")).replace("\n", " ")[:240],
                "notes": row.get("notes"),
            }
        )
    return summaries


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _write_confusion_matrix(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gold", "prediction", "count"])
        writer.writeheader()
        for row in rows:
            item = _mapping(row)
            writer.writerow(
                {
                    "gold": item.get("gold", ""),
                    "prediction": item.get("prediction", ""),
                    "count": item.get("count", 0),
                }
            )


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _format_value(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.6f}"
    if value is None:
        return "N/A"
    return str(value)
