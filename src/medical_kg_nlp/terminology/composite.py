"""Deterministic composition of recognition and normalization repositories."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository

__all__ = ["CompositeTerminologyRepository"]


@dataclass(frozen=True)
class CompositeTerminologyRepository:
    """Query repositories in priority order and deduplicate by concept ID."""

    repositories: tuple[TerminologyRepository, ...]

    def __post_init__(self) -> None:
        if not self.repositories:
            raise ValueError("At least one terminology repository is required")

    def get_by_concept_id(self, concept_id: str) -> ConceptEntry | None:
        for repository in self.repositories:
            entry = repository.get_by_concept_id(concept_id)
            if entry is not None:
                return entry
        return None

    def get_by_code(self, code_system: CodeSystem, code: str) -> ConceptEntry | None:
        for repository in self.repositories:
            entry = repository.get_by_code(code_system, code)
            if entry is not None:
                return entry
        return None

    def exact_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._query(
            lambda repository: repository.exact_lookup(
                mention,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=limit,
            ),
            limit,
        )

    def toneless_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._query(
            lambda repository: repository.toneless_lookup(
                mention,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=limit,
            ),
            limit,
        )

    def search(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._query(
            lambda repository: repository.search(
                mention,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=limit,
            ),
            limit,
        )

    def _query(
        self,
        query: Callable[[TerminologyRepository], list[ConceptEntry]],
        limit: int,
    ) -> list[ConceptEntry]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        output: list[ConceptEntry] = []
        seen: set[str] = set()
        for repository in self.repositories:
            for entry in query(repository):
                if entry.concept_id in seen:
                    continue
                seen.add(entry.concept_id)
                output.append(entry)
                if len(output) == limit:
                    return output
        return output
