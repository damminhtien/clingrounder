"""Internal-schema evaluation command."""

from __future__ import annotations

import argparse
import json

from clingrounder.datasets.synthetic_adapter import SyntheticDatasetAdapter
from clingrounder.evaluation.end_to_end_metrics import evaluate_predictions
from clingrounder.evaluation.error_analysis import write_error_analysis

__all__ = ["evaluate"]


def evaluate(args: argparse.Namespace) -> int:
    """Evaluate predictions and write a focused error-analysis CSV."""

    adapter = SyntheticDatasetAdapter()
    gold = adapter.load_gold(args.gold)
    predictions = adapter.load_gold(args.pred)
    metrics = evaluate_predictions(gold, predictions)
    write_error_analysis(gold, predictions, args.error_analysis)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
