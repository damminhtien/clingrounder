from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match


class DictionaryStore:
    def __init__(self, entries: list[ConceptEntry]) -> None:
        self.entries = entries
        self.by_concept_id = {entry.concept_id: entry for entry in entries}
        self.alias_index: dict[str, list[ConceptEntry]] = defaultdict(list)
        self.toneless_alias_index: dict[str, list[ConceptEntry]] = defaultdict(list)
        for entry in entries:
            for alias in entry.all_names:
                self.alias_index[normalize_for_match(alias)].append(entry)
                self.toneless_alias_index[normalize_for_match(alias, strip_diacritics=True)].append(entry)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "DictionaryStore":
        entries: list[ConceptEntry] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                entries.append(
                    ConceptEntry(
                        concept_id=str(row["concept_id"]),
                        code=row.get("code"),
                        code_system=CodeSystem(row["code_system"]),
                        canonical_name=str(row["canonical_name"]),
                        semantic_type=EntityType(row["semantic_type"]),
                        aliases=tuple(str(alias) for alias in row.get("aliases", [])),
                        parents=tuple(str(parent) for parent in row.get("parents", [])),
                        source=str(row.get("source", "")),
                    )
                )
        return cls(entries)

    def exact_lookup(self, mention: str) -> list[ConceptEntry]:
        return list(self.alias_index.get(normalize_for_match(mention), []))

    def toneless_lookup(self, mention: str) -> list[ConceptEntry]:
        return list(self.toneless_alias_index.get(normalize_for_match(mention, strip_diacritics=True), []))

    def entries_for_type(self, entity_type: EntityType) -> list[ConceptEntry]:
        return [entry for entry in self.entries if entry.semantic_type == entity_type]

    def aliases_for_ner(self) -> list[tuple[str, ConceptEntry]]:
        aliases: list[tuple[str, ConceptEntry]] = []
        for entry in self.entries:
            for alias in entry.all_names:
                aliases.append((alias, entry))
        return sorted(aliases, key=lambda item: len(item[0]), reverse=True)

