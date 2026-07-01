#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.context.cue_loader import load_assertion_cues
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.io import read_jsonl, read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate seed medical dictionaries.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = read_yaml(args.config)
    dictionary_path = Path(config["dictionaries"]["seed_dictionary"])
    source_registry = config["dictionaries"].get("source_registry")
    source_ids: set[str] = set()
    if source_registry:
        source_ids = _validate_source_registry(Path(str(source_registry)))
    store = DictionaryStore.from_jsonl(dictionary_path)
    dictionary_source_count = _validate_dictionary_sources(dictionary_path, source_ids)
    alias_count = 0
    alias_table = config["dictionaries"].get("vietnamese_alias_table")
    if alias_table:
        alias_path = Path(str(alias_table))
        alias_count = _validate_alias_table(alias_path, store)
    cue_count = 0
    cue_table = config["dictionaries"].get("assertion_cue_table")
    if cue_table:
        cue_count = _validate_assertion_cue_table(Path(str(cue_table)), source_ids)
    summary = {
        "dictionary": str(dictionary_path),
        "concepts": len(store.entries),
        "aliases": sum(len(entry.all_names) for entry in store.entries),
        "assertion_cues": cue_count,
        "dictionary_source_links": dictionary_source_count,
        "source_registry_entries": len(source_ids),
        "vietnamese_aliases": alias_count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _validate_source_registry(path: Path) -> set[str]:
    if not path.exists():
        raise ValueError(f"source registry not found: {path}")
    payload = read_yaml(path)
    resources = payload.get("resources")
    if not isinstance(resources, list):
        raise ValueError(f"{path}: resources must be a list")
    source_ids: set[str] = set()
    for index, resource in enumerate(resources, start=1):
        if not isinstance(resource, dict):
            raise ValueError(f"{path}:resources[{index}]: expected mapping")
        source_id = str(resource.get("id", "")).strip()
        if not source_id:
            raise ValueError(f"{path}:resources[{index}]: missing id")
        if source_id in source_ids:
            raise ValueError(f"{path}:resources[{index}]: duplicate id {source_id!r}")
        for required_key in ("name", "category", "access", "url", "license", "use"):
            if not str(resource.get(required_key, "")).strip():
                raise ValueError(f"{path}:resources[{index}]: missing {required_key}")
        source_ids.add(source_id)
    return source_ids


def _validate_dictionary_sources(path: Path, known_source_ids: set[str]) -> int:
    if not known_source_ids:
        return 0
    rows = read_jsonl(path)
    link_count = 0
    for index, row in enumerate(rows, start=1):
        source_ids = _row_source_ids(row)
        if not source_ids:
            raise ValueError(f"{path}:{index}: missing source or source_ids")
        unknown = sorted(source_ids - known_source_ids)
        if unknown:
            raise ValueError(f"{path}:{index}: unknown source ids {unknown}")
        link_count += len(source_ids)
    return link_count


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


def _validate_assertion_cue_table(path: Path, known_source_ids: set[str]) -> int:
    if not path.exists():
        return 0
    cues = load_assertion_cues(path)
    for index, cue in enumerate(cues, start=1):
        unknown = sorted(set(cue.source_ids) - known_source_ids)
        if unknown:
            raise ValueError(f"{path}:{index}: unknown source ids {unknown}")
    return len(cues)


def _row_source_ids(row: dict[str, object]) -> set[str]:
    source_ids: set[str] = set()
    raw_source_ids = row.get("source_ids")
    if isinstance(raw_source_ids, str):
        source_ids.add(raw_source_ids)
    elif isinstance(raw_source_ids, list | tuple):
        source_ids.update(str(item) for item in raw_source_ids if str(item).strip())
    source = row.get("source")
    if isinstance(source, str) and source.strip():
        source_ids.add(source.strip())
    return source_ids


if __name__ == "__main__":
    main()
