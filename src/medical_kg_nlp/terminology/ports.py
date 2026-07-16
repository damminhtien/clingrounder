"""Storage-neutral terminology query contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType

__all__ = ["TerminologyRepository"]


class TerminologyRepository(Protocol):
    """Resolve concepts without exposing JSONL, memory, or SQLite internals."""

    def get_by_concept_id(self, concept_id: str) -> ConceptEntry | None: ...

    def get_by_code(self, code_system: CodeSystem, code: str) -> ConceptEntry | None: ...

    def exact_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]: ...
    def toneless_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]: ...

    def search(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]: ...
