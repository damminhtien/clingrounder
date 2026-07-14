from __future__ import annotations
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match


class DictionaryStore:
    def __init__(self, entries: list[ConceptEntry]) -> None:
        self.entries = entries
        self.by_concept_id = {entry.concept_id: entry for entry in entries}
        self.by_code_system_code = {
            (entry.code_system, entry.code): entry
            for entry in entries
            if entry.code is not None
        }
        self.alias_index: dict[str, list[ConceptEntry]] = defaultdict(list)
        self.toneless_alias_index: dict[str, list[ConceptEntry]] = defaultdict(list)
        for entry in entries:
            for alias in entry.all_names:
                self.alias_index[normalize_for_match(alias)].append(entry)
                self.toneless_alias_index[normalize_for_match(alias, strip_diacritics=True)].append(entry)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        alias_overlay_path: str | Path | None = None,
    ) -> "DictionaryStore":
        return cls(
            cls.load_entries_jsonl(
                path,
                alias_overlay_path=alias_overlay_path,
            )
        )

    @staticmethod
    def load_entries_jsonl(
        path: str | Path,
        *,
        alias_overlay_path: str | Path | None = None,
    ) -> list[ConceptEntry]:
        """Load concepts without constructing lookup indexes.

        Pipeline assembly often merges multiple terminology files. Keeping loading separate avoids
        building large temporary alias indexes that are immediately discarded by the merged store.
        """
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
                        aliases=_string_tuple(row, "aliases"),
                        official_name_vi=_optional_string(row.get("official_name_vi")),
                        official_name_en=_optional_string(row.get("official_name_en")),
                        synonyms=_string_tuple(row, "synonyms"),
                        abbreviations=_string_tuple(row, "abbreviations"),
                        parents=_parents(row),
                        parent_code=_optional_string(row.get("parent_code")),
                        source=str(row.get("source", "")),
                        rxnorm_id=_optional_string(row.get("rxnorm_id")),
                        ingredient=_optional_string(row.get("ingredient")),
                        brand_name=_optional_string(row.get("brand_name")),
                        generic_name=_optional_string(row.get("generic_name")),
                        dose_form=_optional_string(row.get("dose_form")),
                        rxnorm_tty=_optional_string(row.get("rxnorm_tty")),
                        strength=_optional_string(row.get("strength")),
                        blocked_aliases=_string_tuple(row, "blocked_aliases"),
                    )
                )
        return _apply_alias_overlays(entries, alias_overlay_path)

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


def _string_tuple(row: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = row.get(key, [])
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list | tuple):
        raise ValueError(f"Expected string array for {key!r}.")
    return tuple(str(item) for item in value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _parents(row: Mapping[str, Any]) -> tuple[str, ...]:
    parents = list(_string_tuple(row, "parents"))
    parent_code = _optional_string(row.get("parent_code"))
    if parent_code is not None and parent_code not in parents:
        parents.append(parent_code)
    return tuple(parents)


def _apply_alias_overlays(
    entries: list[ConceptEntry],
    alias_overlay_path: str | Path | None,
) -> list[ConceptEntry]:
    if alias_overlay_path is None:
        return entries
    path = Path(alias_overlay_path)
    if not path.exists():
        return entries
    by_concept_id = {entry.concept_id: entry for entry in entries}
    aliases_by_concept_id: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object.")
            target_concept_id = str(row.get("target_concept_id", "")).strip()
            alias = str(row.get("alias", "")).strip()
            if not target_concept_id or not alias:
                raise ValueError(f"{path}:{line_number}: target_concept_id and alias are required.")
            if target_concept_id not in by_concept_id:
                continue
            aliases_by_concept_id[target_concept_id].append(alias)
    if not aliases_by_concept_id:
        return entries

    updated: list[ConceptEntry] = []
    for entry in entries:
        overlay_aliases = aliases_by_concept_id.get(entry.concept_id)
        if not overlay_aliases:
            updated.append(entry)
            continue
        existing = {alias.casefold().strip() for alias in entry.all_names}
        aliases = list(entry.aliases)
        for alias in overlay_aliases:
            key = alias.casefold().strip()
            if key and key not in existing:
                aliases.append(alias)
                existing.add(key)
        updated.append(replace(entry, aliases=tuple(aliases)))
    return updated
