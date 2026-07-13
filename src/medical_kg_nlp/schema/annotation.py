from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType, RelationType


MEDICATION_COMPONENT_KINDS = frozenset(
    {
        "strength",
        "dosage",
        "route",
        "frequency",
        "duration",
        "dose_form",
        "transition",
        "context",
    }
)


@dataclass(frozen=True)
class MedicationComponent:
    kind: str
    span: tuple[int, int]

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "span": [self.span[0], self.span[1]]}


@dataclass(frozen=True)
class MedicationMention:
    drug_span: tuple[int, int]
    full_span: tuple[int, int]
    components: tuple[MedicationComponent, ...] = ()

    def validate_offsets(self, source_text: str, entity_span: tuple[int, int]) -> None:
        if self.drug_span != entity_span:
            raise ValueError("Medication drug_span must equal the entity span.")
        start, end = self.full_span
        if start != entity_span[0] or end < entity_span[1] or end > len(source_text):
            raise ValueError(f"Invalid medication full_span {self.full_span}.")
        for component in self.components:
            component_start, component_end = component.span
            if component.kind not in MEDICATION_COMPONENT_KINDS:
                raise ValueError(f"Unknown medication component kind {component.kind!r}.")
            if not start <= component_start < component_end <= end:
                raise ValueError(f"Invalid medication component span {component.span}.")

    def to_json(self) -> dict[str, Any]:
        return {
            "drug_span": [self.drug_span[0], self.drug_span[1]],
            "full_span": [self.full_span[0], self.full_span[1]],
            "components": [component.to_json() for component in self.components],
        }


@dataclass(frozen=True)
class CandidateConcept:
    code_system: CodeSystem
    code: str | None
    name: str
    score: float
    concept_id: str | None = None
    source: str | None = None
    matched_alias: str | None = None
    qualified: bool = False
    qualification_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "code_system": self.code_system.value,
            "code": self.code,
            "name": self.name,
            "score": round(self.score, 6),
            "source": self.source,
            "matched_alias": self.matched_alias,
            "qualified": self.qualified,
            "qualification_reason": self.qualification_reason,
        }


@dataclass
class AssertionFeatures:
    negated: bool = False
    historical: bool = False
    family: bool = False
    possible: bool = False
    conditional: bool = False
    planned: bool = False
    resolved: bool = False

    @classmethod
    def from_statuses(cls, statuses: set[AssertionStatus]) -> "AssertionFeatures":
        return cls(
            negated=AssertionStatus.NEGATED in statuses,
            historical=AssertionStatus.HISTORICAL in statuses,
            family=AssertionStatus.FAMILY in statuses,
            possible=AssertionStatus.POSSIBLE in statuses,
            conditional=AssertionStatus.CONDITIONAL in statuses,
            planned=AssertionStatus.PLANNED in statuses,
            resolved=AssertionStatus.RESOLVED in statuses,
        )

    def statuses(self) -> tuple[AssertionStatus, ...]:
        values = (
            (self.family, AssertionStatus.FAMILY),
            (self.negated, AssertionStatus.NEGATED),
            (self.historical, AssertionStatus.HISTORICAL),
            (self.planned, AssertionStatus.PLANNED),
            (self.resolved, AssertionStatus.RESOLVED),
            (self.conditional, AssertionStatus.CONDITIONAL),
            (self.possible, AssertionStatus.POSSIBLE),
        )
        return tuple(status for enabled, status in values if enabled)

    def primary(self) -> AssertionStatus:
        statuses = self.statuses()
        return statuses[0] if statuses else AssertionStatus.PRESENT

    def to_json(self) -> dict[str, bool]:
        return {
            "negated": self.negated,
            "historical": self.historical,
            "family": self.family,
            "possible": self.possible,
            "conditional": self.conditional,
            "planned": self.planned,
            "resolved": self.resolved,
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
    assertion_features: AssertionFeatures = field(default_factory=AssertionFeatures)
    medication_mention: MedicationMention | None = None

    def validate_offsets(self, source_text: str) -> None:
        start, end = self.span
        if start < 0 or end < start or end > len(source_text):
            raise ValueError(f"Invalid span {self.span} for entity {self.id}")
        if source_text[start:end] != self.text:
            raise ValueError(
                f"Offset mismatch for {self.id}: expected {self.text!r}, "
                f"got {source_text[start:end]!r}"
            )
        if self.medication_mention is not None:
            self.medication_mention.validate_offsets(source_text, self.span)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "span": [self.span[0], self.span[1]],
            "text": self.text,
            "normalized_text": self.normalized_text,
            "type": self.type.value,
            "assertion": self.assertion.value,
            "assertion_features": self.assertion_features.to_json(),
            "code_system": self.code_system.value,
            "code": self.code,
            "confidence": round(self.confidence, 6),
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }
        if self.medication_mention is not None:
            payload["medication_mention"] = self.medication_mention.to_json()
        return payload


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
