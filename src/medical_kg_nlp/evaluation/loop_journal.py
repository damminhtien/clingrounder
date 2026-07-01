from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_experiment_journal(
    loop_report: dict[str, Any],
    journal_dir: str | Path,
    *,
    run_output_dir: str | Path,
) -> None:
    path = Path(journal_dir)
    path.mkdir(parents=True, exist_ok=True)
    entry = build_journal_entry(loop_report, run_output_dir=run_output_dir)
    log_path = path / "experiments.jsonl"
    _append_jsonl(log_path, entry)

    entries = [*_load_jsonl(log_path)]
    index = build_experiment_index(entries)
    _write_json(path / "experiment_index.json", index)
    _write_json(path / "experiment_memory.json", build_experiment_memory(index))
    (path / "experiment_notebook.md").write_text(render_experiment_notebook(index), encoding="utf-8")


def build_journal_entry(loop_report: dict[str, Any], *, run_output_dir: str | Path) -> dict[str, Any]:
    experiment = _mapping(loop_report["experiment_log"])
    decision = _mapping(loop_report["decision"])
    next_experiment = _mapping(loop_report["next_experiment"])
    top_errors = _dict_list(loop_report["top_errors"])
    run_path = str(Path(run_output_dir))
    return {
        "schema_version": "experiment-journal.v1",
        "id": experiment.get("id"),
        "date": experiment.get("date"),
        "owner": experiment.get("owner"),
        "module": experiment.get("module"),
        "hypothesis": experiment.get("hypothesis"),
        "change": experiment.get("change", []),
        "baseline": experiment.get("baseline"),
        "dataset": experiment.get("dataset", {}),
        "decision": decision.get("decision"),
        "primary_metric": decision.get("primary_metric"),
        "primary_before": decision.get("primary_before"),
        "primary_after": decision.get("primary_after"),
        "primary_delta": decision.get("primary_delta"),
        "blocking_issues": decision.get("blocking_issues"),
        "top_errors": top_errors[:5],
        "next": next_experiment,
        "notes": experiment.get("notes", []),
        "artifacts": {
            "run_dir": run_path,
            "decision": str(Path(run_path) / "decision.md"),
            "agent_compact": str(Path(run_path) / "agent_compact.md"),
            "agent_actions": str(Path(run_path) / "agent_actions.jsonl"),
            "loop_report": str(Path(run_path) / "loop_report.json"),
            "top_error_cases": str(Path(run_path) / "top_error_cases.md"),
        },
        "memory_hint": _memory_hint(decision.get("decision"), top_errors),
    }


def build_experiment_index(entries: list[dict[str, Any]]) -> dict[str, Any]:
    latest_by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        experiment_id = str(entry.get("id", ""))
        if experiment_id:
            latest_by_id[experiment_id] = entry
    latest = sorted(
        latest_by_id.values(),
        key=lambda row: (str(row.get("date", "")), str(row.get("id", ""))),
        reverse=True,
    )
    return {
        "schema_version": "experiment-index.v1",
        "total_log_rows": len(entries),
        "unique_experiments": len(latest),
        "by_decision": _counter_dict(str(row.get("decision", "")) for row in latest),
        "by_module": _counter_dict(str(row.get("module", "")) for row in latest),
        "latest": latest,
    }


def build_experiment_memory(index: dict[str, Any]) -> dict[str, Any]:
    latest = _dict_list(index.get("latest", []))
    return {
        "schema_version": "experiment-memory.v1",
        "summary": {
            "unique_experiments": index.get("unique_experiments", 0),
            "by_decision": index.get("by_decision", {}),
            "by_module": index.get("by_module", {}),
        },
        "reuse": [_memory_row(row) for row in latest if row.get("decision") in {"keep", "baseline"}],
        "avoid": [_memory_row(row) for row in latest if row.get("decision") == "revert"],
        "refine": [_memory_row(row) for row in latest if row.get("decision") == "refine"],
    }


def render_experiment_notebook(index: dict[str, Any]) -> str:
    latest = _dict_list(index.get("latest", []))
    lines = [
        "# Experiment Notebook",
        "",
        f"- Unique experiments: {index.get('unique_experiments', 0)}",
        f"- Total log rows: {index.get('total_log_rows', 0)}",
        "",
        "## Decision Counts",
        "",
    ]
    for decision, count in sorted(_mapping(index.get("by_decision", {})).items()):
        lines.append(f"- {decision or 'UNKNOWN'}: {count}")
    lines.extend(["", "## Experiments", ""])
    if not latest:
        lines.append("- No experiments recorded.")
    for row in latest:
        lines.extend(_render_experiment_row(row))
    return "\n".join(lines) + "\n"


def _render_experiment_row(row: dict[str, Any]) -> list[str]:
    lines = [
        f"### {row.get('id')} - {row.get('decision')}",
        "",
        f"- Date: {row.get('date')}",
        f"- Module: {row.get('module')}",
        f"- Hypothesis: {row.get('hypothesis')}",
        f"- Primary metric: {row.get('primary_metric')}",
        f"- Before: {_format_value(row.get('primary_before'))}",
        f"- After: {_format_value(row.get('primary_after'))}",
        f"- Delta: {_format_value(row.get('primary_delta'))}",
        f"- Memory hint: {row.get('memory_hint')}",
    ]
    artifacts = _mapping(row.get("artifacts", {}))
    if artifacts:
        lines.append(f"- Artifacts: {artifacts.get('run_dir')}")
    top_errors = _dict_list(row.get("top_errors", []))
    if top_errors:
        lines.append("- Top errors:")
        for error in top_errors[:3]:
            lines.append(f"  - {error.get('error_type')}: {error.get('count')}")
    lines.append("")
    return lines


def _memory_hint(decision: Any, top_errors: list[dict[str, Any]]) -> str:
    target = top_errors[0]["error_type"] if top_errors else "no structured error"
    if decision == "keep":
        return f"Reuse this change pattern when {target} appears again."
    if decision == "baseline":
        return "Use this run as a valid baseline for future comparisons."
    if decision == "revert":
        return f"Avoid this change pattern; it did not produce a valid improvement for {target}."
    if decision == "refine":
        return f"Useful evidence but not decisive; refine before reusing for {target}."
    return "Keep as experiment evidence."


def _memory_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "date": row.get("date"),
        "module": row.get("module"),
        "decision": row.get("decision"),
        "hypothesis": row.get("hypothesis"),
        "primary_metric": row.get("primary_metric"),
        "primary_delta": row.get("primary_delta"),
        "memory_hint": row.get("memory_hint"),
        "artifacts": row.get("artifacts"),
    }


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "UNKNOWN"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


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
