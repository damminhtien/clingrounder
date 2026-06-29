from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType, RelationType


@dataclass(frozen=True)
class CandidateConcept:
    code_system: CodeSystem
    code: str | None
    name: str
    score: float
    concept_id: str | None = None
    source: str | None = None
    matched_alias: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "code_system": self.code_system.value,
            "code": self.code,
            "name": self.name,
            "score": round(self.score, 6),
            "source": self.source,
            "matched_alias": self.matched_alias,
        }


@dataclass
class EntityAnnotation:
    id: str
    span: tuple[int, int]
    text: str
    normalized_text: str
    type: EntityType
    assertion: AssertionStatus = AssertionStatus.UNKNOWN
    code_system: CodeSystem = CodeSystem.NONE
    code: str | None = None
    confidence: float = 0.0
    candidates: list[CandidateConcept] = field(default_factory=list)

    def validate_offsets(self, source_text: str) -> None:
        start, end = self.span
        if start < 0 or end < start or end > len(source_text):
            raise ValueError(f"Invalid span {self.span} for entity {self.id}")
        if source_text[start:end] != self.text:
            raise ValueError(
                f"Offset mismatch for {self.id}: expected {self.text!r}, "
                f"got {source_text[start:end]!r}"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "span": [self.span[0], self.span[1]],
            "text": self.text,
            "normalized_text": self.normalized_text,
            "type": self.type.value,
            "assertion": self.assertion.value,
            "code_system": self.code_system.value,
            "code": self.code,
            "confidence": round(self.confidence, 6),
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }


@dataclass
class RelationAnnotation:
    id: str
    head: str
    tail: str
    type: RelationType
    confidence: float
    evidence_span: tuple[int, int] | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "head": self.head,
            "tail": self.tail,
            "type": self.type.value,
            "confidence": round(self.confidence, 6),
        }
        if self.evidence_span is not None:
            payload["evidence_span"] = [self.evidence_span[0], self.evidence_span[1]]
        return payload

