#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.evaluation.manual_gold import (
    compare_manual_gold_gate,
    evaluate_manual_gold,
    load_phase1_directory,
    write_manual_gold_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate flat Phase 1 output against reviewed manual gold.")
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline", default="data/manual_gold/entity_only_baseline.json")
    args = parser.parse_args()

    report = evaluate_manual_gold(
        load_phase1_directory(args.gold_dir),
        load_phase1_directory(args.pred_dir),
    )
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        report["gate"] = compare_manual_gold_gate(report, baseline)
        report["baseline"] = str(baseline_path)
    write_manual_gold_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "all": report["splits"]["all"]["metrics"],
                "holdout": report["splits"]["holdout"]["metrics"],
                "gate": report.get("gate"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
