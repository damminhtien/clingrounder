#!/usr/bin/env python
"""Build a bounded same-concept synonym-pair training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.training.terminology_pairs import (
    SynonymPairMode,
    TerminologyPairConfig,
    build_terminology_synonym_pairs,
    write_terminology_pair_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in SynonymPairMode],
        default=SynonymPairMode.CANONICAL_TO_ALIAS.value,
    )
    parser.add_argument("--max-names-per-concept", type=int, default=16)
    parser.add_argument("--max-pairs-per-concept", type=int, default=32)
    parser.add_argument("--exclude-abbreviations", action="store_true")
    args = parser.parse_args()

    source_paths = tuple(Path(value) for value in args.source)
    entries = [
        entry
        for source_path in source_paths
        for entry in DictionaryStore.load_entries_jsonl(source_path)
    ]
    config = TerminologyPairConfig(
        mode=SynonymPairMode(args.mode),
        max_names_per_concept=args.max_names_per_concept,
        max_pairs_per_concept=args.max_pairs_per_concept,
        include_abbreviations=not args.exclude_abbreviations,
    )
    pairs = build_terminology_synonym_pairs(entries, config=config)
    report = write_terminology_pair_dataset(
        pairs,
        args.output,
        config=config,
        source_fingerprints={
            str(path): _sha256_file(path) for path in source_paths
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
