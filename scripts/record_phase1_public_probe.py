#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.benchmarks.phase1.phase1_probe_gate import (
    append_public_probe_journal,
    evaluate_public_probe_promotion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and journal a public Phase 1 probe.")
    parser.add_argument("--baseline", required=True, help="Baseline grader JSON payload.")
    parser.add_argument("--trial", required=True, help="Trial grader JSON payload.")
    parser.add_argument("--module", choices=("entity", "assertion", "candidate"), required=True)
    parser.add_argument("--probe-name", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--journal-dir", default="outputs/loops/journal")
    parser.add_argument("--policy-diff", help="Optional JSON object or JSON file.")
    parser.add_argument("--non-target-tolerance", type=float, default=0.0001)
    parser.add_argument("--minimum-assertion-gain", type=float, default=0.5)
    parser.add_argument("--minimum-candidate-gain", type=float, default=0.5)
    args = parser.parse_args()

    baseline = _load_json(args.baseline)
    trial = _load_json(args.trial)
    policy_diff = _load_json_or_object(args.policy_diff) if args.policy_diff else {}
    gate = evaluate_public_probe_promotion(
        baseline,
        trial,
        module=args.module,
        minimum_assertion_gain=args.minimum_assertion_gain,
        minimum_candidate_gain=args.minimum_candidate_gain,
        non_target_tolerance=args.non_target_tolerance,
    )
    record = append_public_probe_journal(
        gate,
        args.journal_dir,
        probe_name=args.probe_name,
        artifact_path=args.artifact,
        policy_diff=policy_diff,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


def _load_json(path: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_json_or_object(value: str) -> dict[str, object]:
    path = Path(value)
    if path.exists():
        return _load_json(value)
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--policy-diff must be a JSON object.")
    return payload


if __name__ == "__main__":
    main()
