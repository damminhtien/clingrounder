from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType, RelationType


MEDICATION_COMPONENT_KINDS = frozenset(
    {
        "strength",
        "administered_dose",
        "dosage",
        "route",
        "frequency",
        "duration",
        "dose_form",
        "release",
        "transition",
        "context",
    }
)


@dataclass(frozen=True)
class MedicationComponent:
    kind: str
    span: tuple[int, int]

    def __post_init__(self) -> None:
        if self.kind not in MEDICATION_COMPONENT_KINDS:
            raise ValueError(f"Unknown medication component kind {self.kind!r}.")
        _validate_span(self.span, "medication component")

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "span": [self.span[0], self.span[1]]}


@dataclass(frozen=True)
class MedicationMention:
    drug_span: tuple[int, int]
    full_span: tuple[int, int]
    components: tuple[MedicationComponent, ...] = ()

    def __post_init__(self) -> None:
        _validate_span(self.drug_span, "medication drug")
        _validate_span(self.full_span, "medication full")
        if self.full_span[0] != self.drug_span[0] or self.full_span[1] < self.drug_span[1]:
            raise ValueError("Medication full_span must contain drug_span and share its start")

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
    retrieval_score: float
    emit_probability: float
    concept_id: str
    source: str
    evidence_sources: tuple[str, ...]
    matched_alias: str
    qualified: bool
    qualification_reason: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("retrieval_score", self.retrieval_score),
            ("emit_probability", self.emit_probability),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and between 0 and 1")
        for field_name, text_value in (
            ("concept_id", self.concept_id),
            ("source", self.source),
            ("matched_alias", self.matched_alias),
            ("qualification_reason", self.qualification_reason),
        ):
            if not text_value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if not self.evidence_sources or any(not item.strip() for item in self.evidence_sources):
            raise ValueError("evidence_sources must contain non-empty source names")
        if self.source not in self.evidence_sources:
            raise ValueError("source must be included in evidence_sources")
        if self.code_system is CodeSystem.NONE:
            if self.code is not None:
                raise ValueError("CodeSystem.NONE requires a null candidate code")
        elif self.code is None or not self.code.strip():
            raise ValueError("A non-NONE candidate code system requires a non-empty code")

    def to_json(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "code_system": self.code_system.value,
            "code": self.code,
            "name": self.name,
            "retrieval_score": round(self.retrieval_score, 6),
            "emit_probability": round(self.emit_probability, 6),
            "source": self.source,
            "evidence_sources": list(self.evidence_sources),
            "matched_alias": self.matched_alias,
            "qualified": self.qualified,
            "qualification_reason": self.qualification_reason,
        }


@dataclass(frozen=True)
class AssertionEvidence:
    rule_id: str
    assertion: AssertionStatus
    cue: str
    scope: str

    def to_json(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "assertion": self.assertion.value,
            "cue": self.cue,
            "scope": self.scope,
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
    assertion_evidence: tuple[AssertionEvidence, ...] = ()
    medication_mention: MedicationMention | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Entity id must be non-empty")
        if not self.text:
            raise ValueError("Entity text must be non-empty")
        _validate_span(self.span, "entity")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Entity confidence must be finite and between 0 and 1")
        _validate_code_assignment(self.code_system, self.code, "entity")
        concept_ids = [candidate.concept_id for candidate in self.candidates]
        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError("Entity candidates must have unique concept IDs")

    def validate_offsets(self, source_text: str) -> None:
        start, end = self.span
        if not 0 <= start < end <= len(source_text):
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
            "assertion_evidence": [item.to_json() for item in self.assertion_evidence],
            "code_system": self.code_system.value,
            "code": self.code,
            "confidence": round(self.confidence, 6),
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }
        if self.medication_mention is not None:
            payload["medication_mention"] = self.medication_mention.to_json()
        return payload


@dataclass(frozen=True)
class AmbiguousEntityProposal:
    """Raw-span proposal whose dictionary evidence supports more than one type.

    This is not a final entity and therefore cannot be linked or exported. Hybrid extractors may
    use it as supporting evidence when an independent model resolves one of the candidate types.
    """

    span: tuple[int, int]
    text: str
    normalized_text: str
    candidate_types: tuple[EntityType, ...]
    concept_ids: tuple[str, ...]
    confidence: float
    source: str = "dictionary"

    def __post_init__(self) -> None:
        if len(self.candidate_types) < 2:
            raise ValueError("Ambiguous proposals require at least two candidate types")
        if tuple(sorted(set(self.candidate_types), key=lambda item: item.value)) != (
            self.candidate_types
        ):
            raise ValueError("Ambiguous proposal candidate types must be unique and sorted")
        if not self.concept_ids or tuple(sorted(set(self.concept_ids))) != self.concept_ids:
            raise ValueError("Ambiguous proposal concept IDs must be non-empty, unique, and sorted")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Ambiguous proposal confidence must be finite and between 0 and 1")
        if not self.source:
            raise ValueError("Ambiguous proposal source must be non-empty")

    def validate_offsets(self, source_text: str) -> None:
        start, end = self.span
        if not 0 <= start < end <= len(source_text):
            raise ValueError(f"Invalid ambiguous proposal span {self.span}")
        if source_text[start:end] != self.text:
            raise ValueError(
                f"Ambiguous proposal offset mismatch: expected {self.text!r}, "
                f"got {source_text[start:end]!r}"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "span": [self.span[0], self.span[1]],
            "text": self.text,
            "normalized_text": self.normalized_text,
            "candidate_types": [item.value for item in self.candidate_types],
            "concept_ids": list(self.concept_ids),
            "confidence": round(self.confidence, 6),
            "source": self.source,
        }


@dataclass(frozen=True)
class EntityExtractionResult:
    """Final entities plus non-exportable proposals retained for arbitration."""

    entities: tuple[EntityAnnotation, ...]
    ambiguous_proposals: tuple[AmbiguousEntityProposal, ...] = ()


@dataclass(frozen=True)
class RelationEvidence:
    """Explain why a relation was emitted.

    ``support_score`` is evidence strength, not a calibrated probability unless a
    calibrated relation model explicitly supplies it.
    """

    source: str
    rule_id: str | None = None
    evidence_span: tuple[int, int] | None = None
    support_score: float = 0.0
    provenance: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Relation evidence source must be non-empty")
        if not math.isfinite(self.support_score) or not 0.0 <= self.support_score <= 1.0:
            raise ValueError("Relation evidence support_score must be between 0 and 1")
        if self.evidence_span is not None:
            _validate_span(self.evidence_span, "relation evidence")


@dataclass
class RelationAnnotation:
    id: str
    head: str
    tail: str
    type: RelationType
    confidence: float
    evidence_span: tuple[int, int] | None = None
    evidence: RelationEvidence | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.head.strip() or not self.tail.strip():
            raise ValueError("Relation id, head, and tail must be non-empty")
        if self.head == self.tail:
            raise ValueError("Relation head and tail must identify different entities")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Relation confidence must be finite and between 0 and 1")
        if self.evidence_span is not None:
            _validate_span(self.evidence_span, "relation evidence")

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
        if self.evidence is not None:
            evidence_payload: dict[str, Any] = {
                "source": self.evidence.source,
                "rule_id": self.evidence.rule_id,
                "support_score": round(self.evidence.support_score, 6),
                "provenance": self.evidence.provenance,
            }
            if self.evidence.evidence_span is not None:
                evidence_payload["evidence_span"] = list(self.evidence.evidence_span)
            payload["evidence"] = evidence_payload
        return payload


def _validate_span(span: tuple[int, int], label: str) -> None:
    start, end = span
    if not 0 <= start < end:
        raise ValueError(f"{label} span must satisfy 0 <= start < end: {span}")


def _validate_code_assignment(code_system: CodeSystem, code: str | None, label: str) -> None:
    if code_system is CodeSystem.NONE:
        if code is not None:
            raise ValueError(f"{label} CodeSystem.NONE requires a null code")
    elif code is None or not code.strip():
        raise ValueError(f"{label} code system requires a non-empty code")
