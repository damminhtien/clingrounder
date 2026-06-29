from __future__ import annotations
from dataclasses import dataclass

from medical_kg_nlp.schema.types import CodeSystem, EntityType


@dataclass(frozen=True)
class Candidate:
    concept_id: str
    code: str | None
    code_system: CodeSystem
    canonical_name: str
    semantic_type: EntityType
    score: float
    source: str
    matched_alias: str | None = None

