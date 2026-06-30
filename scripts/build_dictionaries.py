#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.io import read_jsonl, read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate seed medical dictionaries.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = read_yaml(args.config)
    dictionary_path = Path(config["dictionaries"]["seed_dictionary"])
    store = DictionaryStore.from_jsonl(dictionary_path)
    alias_count = 0
    alias_table = config["dictionaries"].get("vietnamese_alias_table")
    if alias_table:
        alias_path = Path(str(alias_table))
        alias_count = _validate_alias_table(alias_path, store)
    summary = {
        "dictionary": str(dictionary_path),
        "concepts": len(store.entries),
        "aliases": sum(len(entry.all_names) for entry in store.entries),
        "vietnamese_aliases": alias_count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _validate_alias_table(path: Path, store: DictionaryStore) -> int:
    if not path.exists():
        return 0
    rows = read_jsonl(path)
    for index, row in enumerate(rows, start=1):
        target_concept_id = str(row["target_concept_id"])
        entry = store.by_concept_id.get(target_concept_id)
        if entry is None:
            raise ValueError(f"{path}:{index}: unknown target_concept_id {target_concept_id!r}")
        semantic_type = EntityType(str(row["semantic_type"]))
        if entry.semantic_type != semantic_type:
            raise ValueError(
                f"{path}:{index}: semantic_type {semantic_type.value!r} does not match "
                f"{target_concept_id} ({entry.semantic_type.value})."
            )
    return len(rows)


if __name__ == "__main__":
    main()
