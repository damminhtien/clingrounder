#!/usr/bin/env python3
"""Filter Qwen exact-quote proposals through reviewed reusable semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.qwen_semantic_gate import (
    filter_high_precision_qwen_proposals,
)
from medical_kg_nlp.mining.io import load_documents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--profile",
        choices=("reviewed", "strict"),
        default="reviewed",
        help="Use strict to exclude short mentions that require boundary verification.",
    )
    args = parser.parse_args()
    source_text_by_doc = {
        str(document.metadata["source_document_id"]): document.text
        for document in load_documents(Path(args.documents))
    }
    report = filter_high_precision_qwen_proposals(
        args.source,
        source_text_by_doc,
        args.output_dir,
        profile=args.profile,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
