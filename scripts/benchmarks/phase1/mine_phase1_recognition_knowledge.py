#!/usr/bin/env python3
"""Mine reviewed Phase 1 train mentions and benchmark recognition on holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clingrounder.benchmarks.phase1.recognition_mining import (
    Phase1RecognitionMiningConfig,
    run_phase1_recognition_mining,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile train-only Phase 1 recognition knowledge and evaluate holdout."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/input"))
    parser.add_argument("--gold-dir", type=Path, default=Path("data/manual_gold"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/manual_gold/holdout_manifest.json"),
    )
    parser.add_argument(
        "--annotation-policy",
        type=Path,
        default=Path("data/manual_gold/compiled/phase1_annotation_policy.yaml"),
    )
    parser.add_argument(
        "--baseline-recognition",
        type=Path,
        default=Path("data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mining/knowledge"),
    )
    parser.add_argument("--min-f1-gain", type=float, default=0.005)
    parser.add_argument("--min-true-positive-gain", type=int, default=5)
    parser.add_argument("--max-false-positive-increase", type=int, default=5)
    args = parser.parse_args()
    manifest = run_phase1_recognition_mining(
        Phase1RecognitionMiningConfig(
            input_dir=args.input_dir,
            gold_dir=args.gold_dir,
            split_manifest=args.split_manifest,
            annotation_policy=args.annotation_policy,
            baseline_recognition=args.baseline_recognition,
            output_root=args.output_root,
            minimum_exact_f1_gain=args.min_f1_gain,
            minimum_true_positive_gain=args.min_true_positive_gain,
            maximum_false_positive_increase=args.max_false_positive_increase,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if manifest["promotion_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
