#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate seed medical dictionaries.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = read_yaml(args.config)
    dictionary_path = Path(config["dictionaries"]["seed_dictionary"])
    store = DictionaryStore.from_jsonl(dictionary_path)
    summary = {
        "dictionary": str(dictionary_path),
        "concepts": len(store.entries),
        "aliases": sum(len(entry.all_names) for entry in store.entries),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
