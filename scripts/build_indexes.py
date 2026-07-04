#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.utils.io import read_yaml
from medical_kg_nlp.utils.text import normalize_for_match


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight lexical alias index.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = read_yaml(args.config)
    dictionary_path = Path(config["dictionaries"]["seed_dictionary"])
    alias_overlay_path = config["dictionaries"].get("vietnamese_alias_table")
    index_path = Path(config["indexes"]["lexical_index"])
    store = DictionaryStore.from_jsonl(
        dictionary_path,
        alias_overlay_path=Path(str(alias_overlay_path)) if alias_overlay_path else None,
    )
    index: dict[str, list[str]] = {}
    for entry in store.entries:
        for alias in entry.all_names:
            index.setdefault(normalize_for_match(alias), []).append(entry.concept_id)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({"index": str(index_path), "aliases": len(index)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
