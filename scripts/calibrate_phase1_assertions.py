#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.evaluation.manual_gold import load_phase1_directory, manual_gold_split
from medical_kg_nlp.evaluation.phase1_selective_calibration import (
    CandidateCalibrationOptions,
    build_assertion_calibration_report,
    write_assertion_calibration_report,
    write_calibrated_assertion_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Phase 1 assertion evidence with document-fold CV.",
    )
    parser.add_argument("--pred", required=True, help="Internal prediction JSONL.")
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    parser.add_argument("--evidence-map-output")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/assertion_calibration",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-support", type=int, default=5)
    parser.add_argument("--minimum-train-support", type=int, default=4)
    parser.add_argument("--abstention-margin", type=float, default=0.05)
    parser.add_argument("--maximum-fold-regression", type=float, default=0.02)
    args = parser.parse_args()

    gold = load_phase1_directory(args.gold_dir)
    if args.split != "all":
        gold = {
            document_id: rows
            for document_id, rows in gold.items()
            if manual_gold_split(document_id) == args.split
        }
    report = build_assertion_calibration_report(
        SyntheticDatasetAdapter().load_gold(args.pred),
        gold,
        options=CandidateCalibrationOptions(
            folds=args.folds,
            minimum_support=args.minimum_support,
            minimum_train_support=args.minimum_train_support,
            abstention_margin=args.abstention_margin,
            maximum_fold_regression=args.maximum_fold_regression,
        ),
    )
    write_assertion_calibration_report(report, args.output_dir)
    calibrated_rows = (
        write_calibrated_assertion_map(report, args.evidence_map_output)
        if args.evidence_map_output
        else []
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir)),
                "observation_count": report["coverage"]["observation_count"],
                "split": args.split,
                "calibrated_evidence_count": len(calibrated_rows),
                "recommended_rule_ids": report["recommended_rule_ids"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
