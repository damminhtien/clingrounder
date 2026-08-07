from __future__ import annotations

from typing import Any

from clingrounder.experiments.loop_policy import AGENT_PLAYBOOKS, DEFAULT_AGENT_PLAYBOOK


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
            "For Phase 1, submit flat entity JSON only; relations stay internal unless schema changes.",
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


def build_agent_poll_state(agent_context: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    next_action = actions[0] if actions else {}
    blocking_issues = _float_value(agent_context.get("blocking_issues"))
    status = "blocked_by_validation" if blocking_issues > 0 else "ready_for_agent"
    return {
        "schema_version": "agent-poll.v1",
        "status": status,
        "poll_interval_seconds": 30,
        "next_action_id": next_action.get("id"),
        "next_action_status": next_action.get("status"),
        "action_count": len(actions),
        "decision": agent_context.get("decision"),
        "module": agent_context.get("module"),
        "target_error": agent_context.get("target_error"),
        "primary_metric": agent_context.get("primary_metric"),
        "blocking_issues": blocking_issues,
        "token_strategy": (
            "Read agent_compact.md first, then agent_actions.jsonl. "
            "Open loop_report.json only for raw evidence."
        ),
        "read_order": [
            "agent_poll.json",
            "agent_compact.md",
            "agent_actions.jsonl",
            "top_error_cases.md",
            "loop_report.json",
        ],
        "artifact_paths": {
            "poll": "agent_poll.json",
            "compact": "agent_compact.md",
            "actions": "agent_actions.jsonl",
            "full_report": "loop_report.json",
            "cases": "top_error_cases.md",
            "decision": "decision.md",
        },
        "completion_markers": [
            "focused regression test updated",
            "targeted commands pass",
            "stage-wise metrics regenerated",
            "loop report regenerated",
            "validation_issue_count does not increase",
        ],
    }


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
        lines.extend(_render_action(action))
    return "\n".join(lines) + "\n"


def render_agent_compact_markdown(
    agent_context: dict[str, Any],
    actions: list[dict[str, Any]],
    poll_state: dict[str, Any],
) -> str:
    next_experiment = _mapping(agent_context["next_experiment"])
    action = actions[0] if actions else {}
    evidence = _mapping(action.get("evidence"))
    top_error = _mapping(evidence.get("top_error"))
    lines = [
        "# Agent Compact Context",
        "",
        f"Status: {poll_state['status']}",
        f"Experiment: {agent_context['experiment_id']}",
        f"Decision: {agent_context['decision']}",
        f"Module: {agent_context['module']}",
        f"Target error: {agent_context['target_error'] or 'N/A'}",
        f"Primary metric: {agent_context['primary_metric']}",
        f"Blocking issues: {_format_value(agent_context['blocking_issues'])}",
        "",
        "## Next Action",
        "",
        f"ID: {action.get('id', 'N/A')}",
        f"Title: {action.get('title', 'N/A')}",
        f"Objective: {action.get('objective', next_experiment.get('hypothesis', 'N/A'))}",
        f"Success: {next_experiment.get('success_criteria', 'N/A')}",
        "",
        "## Evidence",
        "",
        f"Top error: {top_error.get('error_type', 'N/A')}",
        f"Count: {top_error.get('count', 'N/A')}",
        f"Priority: {_format_value(top_error.get('priority'))}",
        f"Examples: {evidence.get('example_count', 0)}",
        "",
        "## Read Order",
        "",
    ]
    lines.extend(f"- {item}" for item in _list(poll_state["read_order"]))
    lines.extend(["", "## Commands", ""])
    lines.extend(f"- `{command}`" for command in _list(action.get("commands"))[:3])
    lines.extend(["", "## Stop If", ""])
    lines.extend(f"- {condition}" for condition in _list(action.get("stop_conditions"))[:4])
    return "\n".join(lines) + "\n"


def _render_action(action: dict[str, Any]) -> list[str]:
    lines = [
        f"### {action['id']}: {action['title']}",
        "",
        f"- Status: {action['status']}",
        f"- Module: {action['module']}",
        f"- Target error: {action['target_error'] or 'N/A'}",
        f"- Objective: {action['objective']}",
        "",
        "Recommended files:",
    ]
    lines.extend(f"- {file_path}" for file_path in _list(action["recommended_files"]))
    lines.extend(["", "Commands:"])
    lines.extend(f"- `{command}`" for command in _list(action["commands"]))
    lines.extend(["", "Acceptance criteria:"])
    lines.extend(f"- {criterion}" for criterion in _list(action["acceptance_criteria"]))
    lines.extend(["", "Stop conditions:"])
    lines.extend(f"- {condition}" for condition in _list(action["stop_conditions"]))
    lines.append("")
    return lines


def _agent_action_title(next_experiment: dict[str, Any]) -> str:
    target_error = next_experiment.get("target_error")
    if target_error:
        return f"Reduce {target_error}"
    return "Harden evaluation on a more difficult split"


def _case_summaries(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": row.get("document_id"),
            "stage": row.get("stage"),
            "error_type": row.get("error_type"),
            "span": row.get("span"),
            "text_window": str(row.get("text_window", "")).replace("\n", " ")[:240],
            "notes": row.get("notes"),
        }
        for row in cases
    ]


def _float_value(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value


def _format_value(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.6f}"
    if value is None:
        return "N/A"
    return str(value)
