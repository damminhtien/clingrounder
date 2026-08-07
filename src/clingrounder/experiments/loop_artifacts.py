from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from clingrounder.experiments.loop_journal import write_experiment_journal


def write_loop_engineering_report(
    loop_report: dict[str, Any],
    output_dir: str | Path,
    *,
    journal_dir: str | Path | None = None,
) -> None:
    path = Path(output_dir)
    journal_path = Path(journal_dir) if journal_dir is not None else path.parent / "journal"
    _attach_journal_paths(loop_report, journal_path)
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "loop_report.json", loop_report)
    _write_yaml(path / "experiment_log.yaml", _mapping(loop_report["experiment_log"]))
    _write_json(path / "experiment_log.json", loop_report["experiment_log"])
    _write_confusion_matrix(path / "confusion_matrix.csv", _list(loop_report["context_confusion_matrix"]))
    _write_json(path / "agent_poll.json", _mapping(_mapping(loop_report["agent"])["poll"]))
    _write_jsonl(path / "agent_actions.jsonl", _dict_list(_mapping(loop_report["agent"])["actions"]))
    (path / "decision.md").write_text(render_decision_markdown(loop_report), encoding="utf-8")
    (path / "next_experiment.md").write_text(render_next_experiment_markdown(loop_report), encoding="utf-8")
    (path / "top_error_cases.md").write_text(render_top_error_cases_markdown(loop_report), encoding="utf-8")
    (path / "agent_brief.md").write_text(str(_mapping(loop_report["agent"])["brief"]), encoding="utf-8")
    (path / "agent_compact.md").write_text(str(_mapping(loop_report["agent"])["compact"]), encoding="utf-8")
    write_experiment_journal(loop_report, journal_path, run_output_dir=path)


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
    lines.extend(f"- {change}" for change in _list(experiment["change"]))
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
    lines.extend(f"- {change}" for change in _list(next_experiment["change"]))
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
                f"- {row['error_type']}: count={row['count']}, "
                f"frequency={_format_value(row['frequency'])}, "
                f"priority={_format_value(row['priority'])}, module={row['module']}"
            )
    lines.extend(["", "## Cases", ""])
    if not cases:
        lines.append("- No cases selected.")
    else:
        for index, row in enumerate(cases, start=1):
            lines.extend(_render_error_case(index, row))
    return "\n".join(lines) + "\n"


def _render_error_case(index: int, row: dict[str, Any]) -> list[str]:
    lines = [
        f"### {index}. {row.get('error_type')} [{row.get('document_id')}]",
        "",
        f"- Stage: {row.get('stage')}",
        f"- Span: {row.get('span')}",
        f"- Notes: {row.get('notes')}",
    ]
    window = str(row.get("text_window", "")).replace("\n", " ")
    if window:
        lines.append(f"- Window: {window}")
    lines.append("")
    return lines


def _attach_journal_paths(loop_report: dict[str, Any], journal_dir: Path) -> None:
    agent = _mapping(loop_report.get("agent"))
    poll = _mapping(agent.get("poll"))
    artifact_paths = _mapping(poll.get("artifact_paths"))
    artifact_paths.update(
        {
            "journal_dir": str(journal_dir),
            "journal_log": str(journal_dir / "experiments.jsonl"),
            "journal_index": str(journal_dir / "experiment_index.json"),
            "journal_memory": str(journal_dir / "experiment_memory.json"),
            "journal_notebook": str(journal_dir / "experiment_notebook.md"),
        }
    )
    poll["artifact_paths"] = artifact_paths
    agent["poll"] = poll
    loop_report["agent"] = agent


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
    return [item for item in value if isinstance(item, dict)]


def _format_value(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.6f}"
    if value is None:
        return "N/A"
    return str(value)
