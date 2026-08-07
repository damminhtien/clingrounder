#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clingrounder.experiments.loop_engineer import (
    build_loop_engineering_report,
    write_loop_engineering_report,
)
from clingrounder.utils.run_output import create_hashed_run_dir, path_in_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn pipeline metrics into an experiment decision and next-step loop report.",
    )
    parser.add_argument(
        "--current-report", required=True, help="Current metrics.json from stage report."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for loop-engineering artifacts."
    )
    parser.add_argument(
        "--run-root",
        help="Optional root for hashed run directories. Relative output paths are written under it.",
    )
    parser.add_argument(
        "--run-label", default="loop", help="Label embedded in the hashed run directory."
    )
    parser.add_argument("--experiment-id", required=True, help="Stable experiment id, e.g. N006.")
    parser.add_argument("--module", required=True, help="Target module, e.g. normalization.")
    parser.add_argument("--hypothesis", required=True, help="One concrete experiment hypothesis.")
    parser.add_argument(
        "--change",
        action="append",
        default=[],
        help="One meaningful change. Repeat for multiple lines in the experiment log.",
    )
    parser.add_argument("--baseline-report", help="Optional baseline metrics.json for comparison.")
    parser.add_argument(
        "--journal-dir",
        help="Optional shared experiment journal directory. Defaults to a journal sibling of output-dir.",
    )
    parser.add_argument("--owner", default="", help="Experiment owner.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset label as key=value, e.g. valid=valid_v3_frozen. Repeatable.",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional experiment note. Repeatable.",
    )
    parser.add_argument(
        "--primary-metric",
        default="loop_score",
        help=(
            "Decision metric. Default uses Phase 1 score when present, "
            "otherwise falls back to internal span/linking/context/relation metrics."
        ),
    )
    parser.add_argument("--keep-delta", type=float, default=0.001)
    parser.add_argument("--revert-delta", type=float, default=0.001)
    parser.add_argument(
        "--top-k", type=int, default=30, help="Maximum top errors/cases to include."
    )
    args = parser.parse_args()

    current_report = _read_json(args.current_report)
    baseline_report = _read_json(args.baseline_report) if args.baseline_report else None
    run_output = (
        create_hashed_run_dir(
            args.run_root,
            label=args.run_label,
            inputs=[args.current_report, args.baseline_report or "none", args.experiment_id],
            resolved_config=vars(args),
        )
        if args.run_root
        else None
    )
    loop_report = build_loop_engineering_report(
        current_report,
        baseline_report=baseline_report,
        experiment_id=args.experiment_id,
        module=args.module,
        hypothesis=args.hypothesis,
        changes=args.change or ["No code change recorded."],
        owner=args.owner,
        dataset=_dataset_mapping(args.dataset),
        notes=list(args.note),
        primary_metric=args.primary_metric,
        keep_delta=args.keep_delta,
        revert_delta=args.revert_delta,
        top_k=args.top_k,
    )
    output_dir = path_in_run(args.output_dir, run_output)
    if run_output:
        loop_report["decision"]["run_id"] = run_output.run_id
        loop_report["decision"]["run_dir"] = str(run_output.run_dir)
        loop_report["decision"]["run_manifest"] = str(run_output.manifest_path)
    write_loop_engineering_report(loop_report, output_dir, journal_dir=args.journal_dir)
    print(json.dumps(loop_report["decision"], ensure_ascii=False, indent=2, sort_keys=True))


def _read_json(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def _dataset_mapping(items: list[str]) -> dict[str, str]:
    dataset: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --dataset values as key=value, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Expected non-empty dataset key in {item!r}")
        dataset[key] = value
    return dataset


if __name__ == "__main__":
    main()
