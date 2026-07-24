"""Storage-neutral terminology query contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType

__all__ = ["TerminologyRepository", "TerminologySearchHit"]


@dataclass(frozen=True)
class TerminologySearchHit:
    """One scored lexical match returned by a terminology repository.

    ``score`` is a bounded lexical similarity used for ranking. It is deliberately
    not named confidence or probability: emission calibration belongs to linking,
    where source, entity type, rank, and downstream evidence are also available.
    """

    entry: ConceptEntry
    score: float
    matched_alias: str
    match_kind: str
    lexical_rank: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Terminology search score must be between 0 and 1")
        if not self.matched_alias:
            raise ValueError("Terminology search hits require a matched alias")
        if not self.match_kind:
            raise ValueError("Terminology search hits require a match kind")


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

    def search_scored(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[TerminologySearchHit]: ...
