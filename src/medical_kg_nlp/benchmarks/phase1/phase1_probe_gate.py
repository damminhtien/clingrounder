from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping


ProbeModule = Literal["entity", "assertion", "candidate"]


def evaluate_public_probe_promotion(
    baseline_payload: Mapping[str, Any],
    trial_payload: Mapping[str, Any],
    *,
    module: ProbeModule,
    minimum_assertion_gain: float = 0.5,
    minimum_candidate_gain: float = 0.5,
    non_target_tolerance: float = 0.0001,
) -> dict[str, Any]:
    baseline = _grader_metrics(baseline_payload)
    trial = _grader_metrics(trial_payload)
    deltas = {
        "primary_score": trial["primary_score"] - baseline["primary_score"],
        "wer_reduction": baseline["wer"] - trial["wer"],
        "assertion_gain": trial["j_assertion"] - baseline["j_assertion"],
        "candidate_gain": trial["j_candidates"] - baseline["j_candidates"],
    }
    checks: dict[str, bool] = {"primary_score_increased": deltas["primary_score"] > 0.0}
    if module == "entity":
        checks.update(
            {
                "wer_decreased": deltas["wer_reduction"] > 0.0,
                "assertion_metric_not_regressed": deltas["assertion_gain"] >= -non_target_tolerance,
                "candidate_metric_not_regressed": deltas["candidate_gain"] >= -non_target_tolerance,
            }
        )
    elif module == "assertion":
        checks.update(
            {
                "assertion_gain": deltas["assertion_gain"] >= minimum_assertion_gain,
                "wer_isolated": abs(deltas["wer_reduction"]) <= non_target_tolerance,
                "candidate_metric_isolated": abs(deltas["candidate_gain"]) <= non_target_tolerance,
            }
        )
    elif module == "candidate":
        checks.update(
            {
                "candidate_gain": deltas["candidate_gain"] >= minimum_candidate_gain,
                "wer_isolated": abs(deltas["wer_reduction"]) <= non_target_tolerance,
                "assertion_metric_isolated": abs(deltas["assertion_gain"]) <= non_target_tolerance,
            }
        )
    else:
        raise ValueError(f"Unknown probe module: {module}")
    return {
        "schema_version": "phase1-public-probe-gate.v1",
        "module": module,
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": {key: round(value, 6) for key, value in deltas.items()},
        "baseline": baseline,
        "trial": trial,
        "thresholds": {
            "minimum_assertion_gain": minimum_assertion_gain,
            "minimum_candidate_gain": minimum_candidate_gain,
            "non_target_tolerance": non_target_tolerance,
        },
    }


def append_public_probe_journal(
    gate: Mapping[str, Any],
    journal_dir: str | Path,
    *,
    probe_name: str,
    artifact_path: str | Path,
    policy_diff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    journal = Path(journal_dir)
    journal.mkdir(parents=True, exist_ok=True)
    artifact = Path(artifact_path)
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "probe_name": probe_name,
        "artifact_path": str(artifact),
        "artifact_sha256": _path_sha256(artifact),
        "decision": "keep" if bool(gate.get("passed")) else "reject",
        "policy_diff": dict(policy_diff or {}),
        "gate": dict(gate),
    }
    jsonl_path = journal / "phase1_top10_public_probes.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (journal / "phase1_top10_public_probes.md").write_text(
        _render_probe_journal(records),
        encoding="utf-8",
    )
    return record


def _grader_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics = payload.get("metrics")
    values = metrics if isinstance(metrics, Mapping) else payload
    primary = payload.get("primaryScore", payload.get("primary_score", payload.get("score")))
    if primary is None and isinstance(values, Mapping):
        primary = values.get("primaryScore", values.get("primary_score", values.get("score")))
    raw = {
        "primary_score": primary,
        "wer": values.get("WER", values.get("wer")) if isinstance(values, Mapping) else None,
        "j_assertion": values.get("J_assertion", values.get("j_assertion"))
        if isinstance(values, Mapping)
        else None,
        "j_candidates": values.get("J_candidates", values.get("j_candidates"))
        if isinstance(values, Mapping)
        else None,
    }
    missing = [key for key, value in raw.items() if not isinstance(value, int | float)]
    if missing:
        raise ValueError(f"Grader payload is missing numeric metrics: {missing}")
    return {key: float(value) for key, value in raw.items()}


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        raise ValueError(f"Artifact does not exist: {path}")
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _render_probe_journal(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 1 Top 10 Public Probes",
        "",
        "| Time | Probe | Module | Score delta | Target delta | Decision | SHA |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for record in records:
        gate = record.get("gate", {})
        deltas = gate.get("deltas", {}) if isinstance(gate, Mapping) else {}
        module = str(gate.get("module", "")) if isinstance(gate, Mapping) else ""
        target_key = {
            "entity": "wer_reduction",
            "assertion": "assertion_gain",
            "candidate": "candidate_gain",
        }.get(module, "primary_score")
        lines.append(
            "| {time} | `{probe}` | {module} | {score:.4f} | {target:.4f} | **{decision}** | `{sha}` |".format(
                time=record.get("timestamp", ""),
                probe=record.get("probe_name", ""),
                module=module,
                score=float(deltas.get("primary_score", 0.0)),
                target=float(deltas.get(target_key, 0.0)),
                decision=record.get("decision", ""),
                sha=str(record.get("artifact_sha256", ""))[:12],
            )
        )
    lines.append("")
    return "\n".join(lines)
