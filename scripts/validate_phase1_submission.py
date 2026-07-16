#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.benchmarks.phase1.phase1 import (
    validate_phase1_submission_dir,
    validate_phase1_submission_zip,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Phase 1 output JSON files or official output.zip.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing source TXT files.")
    parser.add_argument("--output-dir", help="Directory containing output JSON files.")
    parser.add_argument("--zip", dest="zip_path", help="Submission zip path to validate.")
    parser.add_argument(
        "--dictionary",
        default="data/dictionaries/seed_concepts.jsonl",
        help="Dictionary JSONL used to validate ICD-10/RxNorm candidate codes.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=100,
        help="Expected JSON files in output.zip.",
    )
    args = parser.parse_args()

    if not args.output_dir and not args.zip_path:
        parser.error("At least one of --output-dir or --zip is required.")

    dictionary = DictionaryStore.from_jsonl(args.dictionary)
    issues = []
    if args.output_dir:
        issues.extend(
            issue.to_json()
            for issue in validate_phase1_submission_dir(
                args.input_dir,
                args.output_dir,
                dictionary=dictionary,
            )
        )
    if args.zip_path:
        issues.extend(
            issue.to_json()
            for issue in validate_phase1_submission_zip(
                args.zip_path,
                input_dir=args.input_dir,
                dictionary=dictionary,
                expected_count=args.expected_count,
            )
        )

    by_kind: dict[str, int] = {}
    for issue in issues:
        kind = str(issue["kind"])
        by_kind[kind] = by_kind.get(kind, 0) + 1
    summary = {
        "valid": not issues,
        "issue_count": len(issues),
        "by_kind": dict(sorted(by_kind.items())),
        "issues": issues[:50],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
