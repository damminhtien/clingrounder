#!/usr/bin/env python
"""Build the persistent terminology index without coupling it to pipeline startup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.terminology import build_terminology_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a versioned read-only SQLite FTS5 terminology index."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Canonical terminology JSONL. Repeat to merge sources in order.",
    )
    parser.add_argument("--output", help="Explicit SQLite path; defaults to content cache.")
    parser.add_argument(
        "--cache-dir",
        default=".cache/medical-kg/terminology",
        help="Content-addressed cache used when --output is omitted.",
    )
    args = parser.parse_args()

    manifest = build_terminology_index(
        tuple(args.source),
        output_path=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(manifest.to_json(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
