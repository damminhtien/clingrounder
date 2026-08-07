#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from clingrounder.benchmarks.phase1.manual_gold_manifest import (
    sync_manual_gold_manifest,
    write_manual_gold_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize Phase 1 manual-gold review manifest coverage and entity counts."
    )
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument("--input-dir", default="data/raw/input")
    parser.add_argument("--manifest", default="data/manual_gold/review_manifest.jsonl")
    parser.add_argument("--status", default="btc_v1_reviewed")
    parser.add_argument("--reviewed-by", default="codex_raw_review_with_pipeline_proposal")
    parser.add_argument("--review-date", default="2026-07-13")
    parser.add_argument(
        "--refresh-candidate-policy",
        action="store_true",
        help="Replace stale per-document code lists with the locked-source policy.",
    )
    args = parser.parse_args()

    rows = sync_manual_gold_manifest(
        args.gold_dir,
        args.input_dir,
        args.manifest,
        status=args.status,
        reviewed_by=args.reviewed_by,
        review_date=args.review_date,
        refresh_candidate_policy=args.refresh_candidate_policy,
    )
    count = write_manual_gold_manifest(rows, args.manifest)
    print(
        json.dumps(
            {"manifest": args.manifest, "document_count": count},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
