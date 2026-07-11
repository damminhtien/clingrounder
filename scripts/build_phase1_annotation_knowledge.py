#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.evaluation.annotation_knowledge import (
    compile_annotation_knowledge,
    write_annotation_knowledge,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile reviewed Phase 1 labels and review notes into a conflict-aware policy."
    )
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument("--manifest", default="data/manual_gold/review_manifest.jsonl")
    parser.add_argument("--output-dir", default="data/manual_gold/compiled")
    parser.add_argument(
        "--strict-document-support",
        type=int,
        default=2,
        help="Minimum distinct reviewed documents required for a strict positive alias.",
    )
    args = parser.parse_args()
    if args.strict_document_support < 1:
        parser.error("--strict-document-support must be at least 1")

    report = compile_annotation_knowledge(
        gold_dir=args.gold_dir,
        manifest_path=args.manifest,
        strict_document_support=args.strict_document_support,
    )
    write_annotation_knowledge(report, args.output_dir)
    summary = report["summary"]
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "reviewed_document_count": summary["reviewed_document_count"],
                "accepted_entity_count": summary["accepted_entity_count"],
                "strict_alias_count": summary["strict_alias_count"],
                "context_required_alias_count": summary["context_required_alias_count"],
                "strict_exclusion_count": summary["strict_exclusion_count"],
                "conflict_count": summary["conflict_count"],
                "conflict_count_by_severity": summary["conflict_count_by_severity"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
