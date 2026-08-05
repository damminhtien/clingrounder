"""Deterministic composition of recognition and normalization repositories."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.ports import (
    TerminologyRepository,
    TerminologySearchHit,
)

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

    def contains(self, code_system: CodeSystem, code: str) -> bool:
        """Return membership from the first release component containing the code."""

        return any(repository.contains(code_system, code) for repository in self.repositories)

    def close(self) -> None:
        """Close child repositories in reverse composition order."""

        seen: set[int] = set()
        for repository in reversed(self.repositories):
            close = getattr(repository, "close", None)
            if callable(close) and id(repository) not in seen:
                seen.add(id(repository))
                close()

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
        return [
            hit.entry
            for hit in self.search_scored(
                mention,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=limit,
            )
        ]

    def search_scored(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[TerminologySearchHit]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        best: dict[str, tuple[TerminologySearchHit, int]] = {}
        for repository_index, repository in enumerate(self.repositories):
            for hit in repository.search_scored(
                mention,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=limit,
            ):
                current = best.get(hit.entry.concept_id)
                if current is None or hit.score > current[0].score:
                    best[hit.entry.concept_id] = (hit, repository_index)
        ordered = sorted(
            best.values(),
            key=lambda item: (
                -item[0].score,
                item[1],
                item[0].entry.code_system.value,
                item[0].entry.code or "",
                item[0].entry.concept_id,
            ),
        )
        return [hit for hit, _ in ordered[:limit]]

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
