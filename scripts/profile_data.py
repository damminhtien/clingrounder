#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clingrounder.evaluation.data_profile import profile_paths, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile clinical NLP data distributions and risks.")
    parser.add_argument("--documents", required=True, help="Source documents JSONL.")
    parser.add_argument("--gold", required=True, help="Gold annotations in internal prediction JSONL.")
    parser.add_argument(
        "--dictionary",
        default="data/dictionaries/seed_concepts.jsonl",
        help="Dictionary JSONL used to compute code coverage.",
    )
    parser.add_argument(
        "--reference-gold",
        help="Optional reference/train gold JSONL for unseen-code overlap analysis.",
    )
    parser.add_argument("--output", help="Optional JSON output path. Prints JSON when omitted.")
    parser.add_argument("--markdown", help="Optional Markdown summary output path.")
    parser.add_argument("--top-k", type=int, default=20, help="Maximum rows per top-list section.")
    args = parser.parse_args()

    profile = profile_paths(
        documents_path=args.documents,
        gold_path=args.gold,
        dictionary_path=args.dictionary,
        reference_gold_path=args.reference_gold,
        top_k=args.top_k,
    )

    payload = json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(profile), encoding="utf-8")


if __name__ == "__main__":
    main()
