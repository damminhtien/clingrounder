"""In-memory terminology repository for reviewed recognition dictionaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.ports import TerminologySearchHit

__all__ = ["InMemoryTerminologyRepository"]


@dataclass(frozen=True)
class InMemoryTerminologyRepository:
    """Adapt a recognition-sized DictionaryStore to the repository contract."""

    store: DictionaryStore

    def get_by_concept_id(self, concept_id: str) -> ConceptEntry | None:
        return self.store.by_concept_id.get(concept_id)

    def get_by_code(self, code_system: CodeSystem, code: str) -> ConceptEntry | None:
        return self.store.by_code_system_code.get((code_system, code))

    def contains(self, code_system: CodeSystem, code: str) -> bool:
        """Check the store's immutable code index without scanning entries."""

        return (code_system, code) in self.store.by_code_system_code

    def exact_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._filter(self.store.exact_lookup(mention), entity_type, code_systems, limit)

    def toneless_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._filter(self.store.toneless_lookup(mention), entity_type, code_systems, limit)

    def search(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        exact = self.exact_lookup(
            mention,
            entity_type=entity_type,
            code_systems=code_systems,
            limit=limit,
        )
        if len(exact) >= limit:
            return exact
        seen = {entry.concept_id for entry in exact}
        toneless = self.toneless_lookup(
            mention,
            entity_type=entity_type,
            code_systems=code_systems,
            limit=limit,
        )
        return [*exact, *(entry for entry in toneless if entry.concept_id not in seen)][:limit]

    def search_scored(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[TerminologySearchHit]:
        exact = self.exact_lookup(
            mention,
            entity_type=entity_type,
            code_systems=code_systems,
            limit=limit,
        )
        hits = [
            TerminologySearchHit(
                entry=entry,
                score=1.0,
                matched_alias=mention,
                match_kind="exact",
            )
            for entry in exact
        ]
        if len(hits) >= limit:
            return hits
        seen = {entry.concept_id for entry in exact}
        hits.extend(
            TerminologySearchHit(
                entry=entry,
                score=0.92,
                matched_alias=mention,
                match_kind="toneless",
            )
            for entry in self.toneless_lookup(
                mention,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=limit,
            )
            if entry.concept_id not in seen
        )
        return hits[:limit]

    @staticmethod
    def _filter(
        entries: list[ConceptEntry],
        entity_type: EntityType | None,
        code_systems: Sequence[CodeSystem] | None,
        limit: int,
    ) -> list[ConceptEntry]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        allowed_systems = set(code_systems) if code_systems is not None else None
        return [
            entry
            for entry in entries
            if (entity_type is None or entry.semantic_type == entity_type)
            and (allowed_systems is None or entry.code_system in allowed_systems)
        ][:limit]
