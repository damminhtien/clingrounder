#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.schema.validator import PredictionValidator
from medical_kg_nlp.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prediction JSONL schema and invariants.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL to validate.")
    parser.add_argument(
        "--documents",
        help="Optional source JSONL with document_id and text for offset/hash checks.",
    )
    parser.add_argument(
        "--dictionary",
        default="data/dictionaries/seed_concepts.jsonl",
        help="Dictionary JSONL used to reject unknown output codes.",
    )
    args = parser.parse_args()

    documents_by_id = _load_documents(args.documents) if args.documents else {}
    dictionary = DictionaryStore.from_jsonl(args.dictionary) if args.dictionary else None
    validator = PredictionValidator(dictionary)

    issue_count = 0
    rows = read_jsonl(args.pred)
    for line_number, row in enumerate(rows, start=1):
        document_id = str(row.get("document_id", ""))
        source_text = documents_by_id.get(document_id)
        _, issues = validator.validate_payload(row, source_text=source_text)
        for issue in issues:
            issue_count += 1
            print(
                f"{args.pred}:{line_number}: {issue.kind} {issue.path}: {issue.message}",
                file=sys.stderr,
            )

    if issue_count:
        print(f"Validation failed with {issue_count} issue(s).", file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated {len(rows)} prediction row(s) with 0 issues.")


def _load_documents(path: str) -> dict[str, str]:
    documents: dict[str, str] = {}
    for row in read_jsonl(path):
        documents[str(row["document_id"])] = str(row["text"])
    return documents


if __name__ == "__main__":
    main()
