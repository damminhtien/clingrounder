from __future__ import annotations
from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore


def load_umls_dictionary(path: str | Path) -> DictionaryStore:
    return DictionaryStore.from_jsonl(path)

