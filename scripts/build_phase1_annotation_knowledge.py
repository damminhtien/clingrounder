#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.benchmarks.phase1.annotation_knowledge import (
    compile_annotation_knowledge,
    write_annotation_knowledge,
)
from medical_kg_nlp.benchmarks.phase1.manual_gold import manual_gold_split
from medical_kg_nlp.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile reviewed Phase 1 labels and review notes into a conflict-aware policy."
    )
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument("--manifest", default="data/manual_gold/review_manifest.jsonl")
    parser.add_argument(
        "--conflict-decisions",
        default="data/manual_gold/conflict_decisions.jsonl",
        help="Concept-level context/type decisions; document ids and absolute spans are forbidden.",
    )
    parser.add_argument("--output-dir", default="data/manual_gold/compiled")
    parser.add_argument(
        "--strict-document-support",
        type=int,
        default=2,
        help="Minimum distinct reviewed documents required for a strict positive alias.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "holdout"),
        default="train",
        help="Compile runtime knowledge from train by default; holdout remains sealed.",
    )
    parser.add_argument(
        "--allow-unresolved-conflicts",
        action="store_true",
        help="Write exploratory reports without failing when the conflict queue is non-empty.",
    )
    args = parser.parse_args()
    if args.strict_document_support < 1:
        parser.error("--strict-document-support must be at least 1")

    manifest_rows = read_jsonl(args.manifest)
    document_ids = None
    if args.split != "all":
        document_ids = [
            str(row.get("document_id", ""))
            for row in manifest_rows
            if manual_gold_split(str(row.get("document_id", ""))) == args.split
        ]
    report = compile_annotation_knowledge(
        gold_dir=args.gold_dir,
        manifest_path=args.manifest,
        strict_document_support=args.strict_document_support,
        document_ids=document_ids,
        conflict_decisions_path=args.conflict_decisions,
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
                "resolved_conflict_count": summary["resolved_conflict_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if summary["conflict_count"] and not args.allow_unresolved_conflicts:
        raise SystemExit(
            "Annotation knowledge contains unresolved conflicts; review policy_conflicts.csv "
            "or rerun explicitly with --allow-unresolved-conflicts."
        )


if __name__ == "__main__":
    main()
