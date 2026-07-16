#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.benchmarks.phase1.manual_gold_convention import (
    audit_manual_gold_convention,
    write_manual_gold_convention_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase 1 manual gold against BTC sample conventions."
    )
    parser.add_argument("--input-dir", default="data/raw/input")
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument(
        "--decisions",
        default="data/manual_gold/convention_decisions.jsonl",
        help="Concept-level reviewed exceptions; document ids and absolute spans are forbidden.",
    )
    parser.add_argument("--output-dir", default="outputs/evaluation/manual_gold_convention")
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when blocking or unresolved review issues are present.",
    )
    args = parser.parse_args()

    report = audit_manual_gold_convention(
        args.input_dir,
        args.gold_dir,
        expected_count=args.expected_count,
        decisions_path=args.decisions,
    )
    write_manual_gold_convention_report(report, args.output_dir)
    print(json.dumps({key: value for key, value in report.items() if key != "issues"}, ensure_ascii=False, indent=2))
    if args.strict and (report["blocking_count"] or report["review_count"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
