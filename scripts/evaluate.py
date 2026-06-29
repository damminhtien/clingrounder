#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.evaluation.end_to_end_metrics import evaluate_predictions
from medical_kg_nlp.evaluation.error_analysis import write_error_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions against internal-schema gold JSONL.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--error-analysis", default="outputs/error_analysis.csv")
    args = parser.parse_args()

    adapter = SyntheticDatasetAdapter()
    gold = adapter.load_gold(args.gold)
    pred = adapter.load_gold(args.pred)
    metrics = evaluate_predictions(gold, pred)
    write_error_analysis(gold, pred, args.error_analysis)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
