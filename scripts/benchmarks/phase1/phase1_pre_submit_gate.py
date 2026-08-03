#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.experiments.loop_engineer import build_loop_engineering_report, write_loop_engineering_report
from medical_kg_nlp.benchmarks.phase1.phase1_submission_analysis import (
    build_phase1_submission_analysis,
    write_phase1_submission_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a Phase 1 output.zip, analyze submission risks, and feed loop-engineer artifacts.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing Phase 1 input TXT files.")
    parser.add_argument("--zip", dest="zip_path", required=True, help="Phase 1 output.zip path.")
    parser.add_argument("--output-dir", required=True, help="Artifact directory for analysis and loop report.")
    parser.add_argument("--dictionary", default="data/dictionaries/seed_concepts.jsonl")
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--journal-dir", default="outputs/loops/journal")
    parser.add_argument("--experiment-id", help="Stable experiment id for the loop journal.")
    parser.add_argument("--module", default="phase1_submission")
    parser.add_argument("--hypothesis", default="Pre-submit gate catches Phase 1 validation and scoring risks before upload.")
    parser.add_argument("--change", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--score", type=float, help="External grader score, when available.")
    parser.add_argument("--wer", type=float, help="External grader WER, when available.")
    parser.add_argument("--j-assertion", type=float, help="External grader J_assertion, when available.")
    parser.add_argument("--j-candidates", type=float, help="External grader J_candidates, when available.")
    parser.add_argument("--num-scored", type=float, help="External grader num_scored, when available.")
    parser.add_argument("--num-records", type=float, help="External grader num_records, when available.")
    parser.add_argument(
        "--no-fail-on-validation",
        action="store_true",
        help="Write artifacts but exit 0 even if validation issues are found.",
    )
    args = parser.parse_args()

    dictionary = DictionaryStore.from_jsonl(args.dictionary)
    report = build_phase1_submission_analysis(
        input_dir=args.input_dir,
        zip_path=args.zip_path,
        dictionary=dictionary,
        expected_count=args.expected_count,
        external_metrics=_external_metrics(args),
    )
    output_dir = Path(args.output_dir)
    write_phase1_submission_analysis(report, output_dir)

    experiment_id = args.experiment_id or f"PHASE1_GATE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    loop_report = build_loop_engineering_report(
        report,
        experiment_id=experiment_id,
        module=args.module,
        hypothesis=args.hypothesis,
        changes=args.change or [f"Validate and analyze {args.zip_path}."],
        dataset={"input": args.input_dir},
        notes=args.note,
        primary_metric="loop_score",
        top_k=20,
    )
    write_loop_engineering_report(loop_report, output_dir, journal_dir=args.journal_dir)

    validation_summary = report["phase1"]["validation_summary"]
    summary = {
        "valid": validation_summary["issue_count"] == 0,
        "issue_count": validation_summary["issue_count"],
        "output_dir": str(output_dir),
        "analysis": str(output_dir / "analysis.md"),
        "report": str(output_dir / "external_grader_report.json"),
        "loop_report": str(output_dir / "loop_report.json"),
        "decision": loop_report["decision"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if validation_summary["issue_count"] and not args.no_fail_on_validation:
        raise SystemExit(1)


def _external_metrics(args: argparse.Namespace) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for arg_name, metric_name in (
        ("score", "score"),
        ("wer", "wer"),
        ("j_assertion", "j_assertion"),
        ("j_candidates", "j_candidates"),
        ("num_scored", "num_scored"),
        ("num_records", "num_records"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            metrics[metric_name] = float(value)
    return metrics


if __name__ == "__main__":
    main()
