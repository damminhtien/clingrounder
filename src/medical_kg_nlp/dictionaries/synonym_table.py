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
    aliases: tuple[str, ...] = ()
    official_name_vi: str | None = None
    official_name_en: str | None = None
    synonyms: tuple[str, ...] = ()
    abbreviations: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    parent_code: str | None = None
    source: str = ""
    rxnorm_id: str | None = None
    ingredient: str | None = None
    brand_name: str | None = None
    generic_name: str | None = None
    dose_form: str | None = None
    rxnorm_tty: str | None = None
    strength: str | None = None
    blocked_aliases: tuple[str, ...] = ()

    @property
    def all_names(self) -> tuple[str, ...]:
        blocked = {alias.lower().strip() for alias in self.blocked_aliases}
        names = (
            self.canonical_name,
            self.official_name_en,
            self.official_name_vi,
            *self.aliases,
            *self.synonyms,
            *self.abbreviations,
            self.ingredient,
            self.brand_name,
            self.generic_name,
        )
        unique: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name is None:
                continue
            normalized = name.strip()
            key = normalized.lower()
            if not normalized or key in blocked or key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return tuple(unique)
