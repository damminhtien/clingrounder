from __future__ import annotations
from dataclasses import dataclass
import math

from clingrounder.schema.types import CodeSystem, EntityType


@dataclass(frozen=True)
class CandidateEvidence:
    source: str
    score: float
    rank: int
    concept_id: str
    matched_alias: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.concept_id.strip():
            raise ValueError("Candidate evidence source and concept_id must be non-empty")
        if self.rank < 0 or not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("Candidate evidence rank/score is invalid")


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

    def __post_init__(self) -> None:
        if not self.concept_id.strip() or not self.canonical_name.strip() or not self.source.strip():
            raise ValueError("Candidate concept_id, canonical_name, and source must be non-empty")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("Candidate score must be finite and between 0 and 1")
        if self.code_system is CodeSystem.NONE:
            if self.code is not None:
                raise ValueError("CodeSystem.NONE requires a null candidate code")
        elif self.code is None or not self.code.strip():
            raise ValueError("A non-NONE candidate code system requires a non-empty code")

    @property
    def sources(self) -> tuple[str, ...]:
        if self.evidence:
            return tuple(item.source for item in self.evidence)
        return (self.source,)
