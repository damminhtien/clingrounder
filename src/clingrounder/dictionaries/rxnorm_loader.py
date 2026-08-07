from __future__ import annotations
from pathlib import Path

from clingrounder.dictionaries.dictionary_store import DictionaryStore


def load_rxnorm_dictionary(path: str | Path) -> DictionaryStore:
    return DictionaryStore.from_jsonl(path)

