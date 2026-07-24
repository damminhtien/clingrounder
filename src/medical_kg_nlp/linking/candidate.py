from __future__ import annotations
from dataclasses import dataclass

from medical_kg_nlp.schema.types import CodeSystem, EntityType


@dataclass(frozen=True)
class CandidateEvidence:
    source: str
    score: float
    rank: int
    concept_id: str
    matched_alias: str | None = None


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
    evidence: tuple[CandidateEvidence, ...] = ()
    reviewed_mapping: bool = False

    @property
    def sources(self) -> tuple[str, ...]:
        if self.evidence:
            return tuple(item.source for item in self.evidence)
        return (self.source,)
