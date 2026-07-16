#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.benchmarks.phase1.manual_gold import (
    build_manual_gold_split_manifest,
    verify_manual_gold_split_manifest,
    write_manual_gold_split_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the deterministic Phase 1 manual-gold holdout split."
    )
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument("--documents", default="data/raw/input")
    parser.add_argument(
        "--output",
        default="data/manual_gold/holdout_manifest.json",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace an existing manifest after intentional corpus changes.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    existed_before = output.exists()
    if existed_before and not args.replace:
        manifest = json.loads(output.read_text(encoding="utf-8"))
        verify_manual_gold_split_manifest(manifest, args.gold_dir, args.documents)
        action = "verified"
    else:
        manifest = build_manual_gold_split_manifest(args.gold_dir, args.documents)
        write_manual_gold_split_manifest(manifest, output)
        action = "replaced" if existed_before else "created"

    print(
        json.dumps(
            {
                "action": action,
                "output": str(output),
                "corpus": manifest["corpus"],
                "train_document_count": manifest["splits"]["train"]["document_count"],
                "holdout_document_count": manifest["splits"]["holdout"]["document_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
