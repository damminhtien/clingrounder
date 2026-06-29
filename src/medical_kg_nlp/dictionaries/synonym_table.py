from __future__ import annotations
from dataclasses import dataclass

from medical_kg_nlp.schema.types import CodeSystem, EntityType


@dataclass(frozen=True)
class ConceptEntry:
    concept_id: str
    code: str | None
    code_system: CodeSystem
    canonical_name: str
    semantic_type: EntityType
    aliases: tuple[str, ...]
    parents: tuple[str, ...] = ()
    source: str = ""

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.canonical_name, *self.aliases)

