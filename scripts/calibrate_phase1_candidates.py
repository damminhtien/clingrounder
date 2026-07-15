#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.evaluation.manual_gold import load_phase1_directory, manual_gold_split
from medical_kg_nlp.evaluation.phase1 import load_reviewed_candidate_map
from medical_kg_nlp.evaluation.phase1_selective_calibration import (
    CandidateCalibrationOptions,
    build_candidate_calibration_report,
    write_candidate_calibration_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate selective Phase 1 candidate emission with document-fold CV.",
    )
    parser.add_argument("--pred", required=True, help="Internal prediction JSONL.")
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    parser.add_argument("--reviewed-map")
    parser.add_argument(
        "--dictionary",
        action="append",
        default=[],
        help="Terminology JSONL used to classify RxNorm TTY and ICD parent/leaf coverage.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/candidate_calibration",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-support", type=int, default=5)
    parser.add_argument("--minimum-train-support", type=int, default=4)
    parser.add_argument("--abstention-margin", type=float, default=0.05)
    parser.add_argument("--maximum-fold-regression", type=float, default=0.02)
    args = parser.parse_args()

    predictions = SyntheticDatasetAdapter().load_gold(args.pred)
    gold = load_phase1_directory(args.gold_dir)
    if args.split != "all":
        gold = {
            document_id: rows
            for document_id, rows in gold.items()
            if manual_gold_split(document_id) == args.split
        }
    reviewed = (
        load_reviewed_candidate_map(args.reviewed_map)
        if args.reviewed_map
        else frozenset()
    )
    report = build_candidate_calibration_report(
        predictions,
        gold,
        reviewed_candidates=reviewed,
        terminology_entries=[
            entry
            for path in args.dictionary
            for entry in DictionaryStore.load_entries_jsonl(path)
        ],
        options=CandidateCalibrationOptions(
            folds=args.folds,
            minimum_support=args.minimum_support,
            minimum_train_support=args.minimum_train_support,
            abstention_margin=args.abstention_margin,
            maximum_fold_regression=args.maximum_fold_regression,
        ),
    )
    write_candidate_calibration_report(report, args.output_dir)
    summary = {
        "output_dir": str(Path(args.output_dir)),
        "observation_count": report["coverage"]["observation_count"],
        "split": args.split,
        "recommended_link_emit_probabilities_by_source": report[
            "recommended_link_emit_probabilities_by_source"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
