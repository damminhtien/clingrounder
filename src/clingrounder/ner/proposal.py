"""Immutable evidence records shared by rule-based entity extractors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from clingrounder.schema.annotation import MedicationMention
from clingrounder.schema.types import EntityType

__all__ = [
    "EntityProposal",
    "ProposalDecision",
    "RuleNerTrace",
]


@dataclass(frozen=True, slots=True)
class EntityProposal:
    """One raw-text span proposed by an independent recognition source.

    A proposal may carry multiple candidate types while evidence is unresolved. Such proposals
    are retained for type arbitration but cannot become final entities until exactly one type
    remains.
    """

    span: tuple[int, int]
    candidate_types: tuple[EntityType, ...]
    source: str
    score: float
    evidence_ids: tuple[str, ...] = ()
    concept_ids: tuple[str, ...] = ()
    features: tuple[tuple[str, str], ...] = ()
    medication_mention: MedicationMention | None = None

    def __post_init__(self) -> None:
        start, end = self.span
        if start < 0 or end <= start:
            raise ValueError(f"Entity proposal requires a non-empty span, got {self.span}")
        expected_types = tuple(sorted(set(self.candidate_types), key=lambda item: item.value))
        if not expected_types or self.candidate_types != expected_types:
            raise ValueError("candidate_types must be non-empty, unique, and sorted")
        if not self.source.strip():
            raise ValueError("Entity proposal source must be non-empty")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("Entity proposal score must be finite and between 0 and 1")
        if tuple(sorted(set(self.evidence_ids))) != self.evidence_ids:
            raise ValueError("evidence_ids must be unique and sorted")
        if tuple(sorted(set(self.concept_ids))) != self.concept_ids:
            raise ValueError("concept_ids must be unique and sorted")
        if tuple(sorted(set(self.features))) != self.features:
            raise ValueError("features must be unique and sorted")

    @property
    def entity_type(self) -> EntityType | None:
        """Return the resolved type, or ``None`` while the proposal remains ambiguous."""

        return self.candidate_types[0] if len(self.candidate_types) == 1 else None

    def feature(self, name: str) -> str | None:
        """Read one immutable feature without exposing a mutable metadata mapping."""

        return next((value for key, value in self.features if key == name), None)

    def validate_offsets(self, source_text: str) -> None:
        """Validate that this proposal references a non-empty raw-text substring."""

        start, end = self.span
        if end > len(source_text):
            raise ValueError(
                f"Proposal span {self.span} exceeds source length {len(source_text)}"
            )
        if not source_text[start:end]:
            raise ValueError(f"Proposal span {self.span} resolves to empty text")
        if self.medication_mention is not None:
            self.medication_mention.validate_offsets(source_text, self.span)

    def to_json(self, source_text: str) -> dict[str, Any]:
        """Serialize evidence with the exact raw substring for audit reports."""

        self.validate_offsets(source_text)
        return {
            "span": [self.span[0], self.span[1]],
            "text": source_text[self.span[0] : self.span[1]],
            "candidate_types": [item.value for item in self.candidate_types],
            "source": self.source,
            "score": round(self.score, 6),
            "evidence_ids": list(self.evidence_ids),
            "concept_ids": list(self.concept_ids),
            "features": dict(self.features),
        }


@dataclass(frozen=True, slots=True)
class ProposalDecision:
    """Auditable accept/reject outcome for one source proposal."""

    span: tuple[int, int]
    source: str
    candidate_types: tuple[EntityType, ...]
    accepted: bool
    reason: str
    selected_type: EntityType | None = None
    competing_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.reason.strip():
            raise ValueError("Proposal decisions require non-empty source and reason")
        if tuple(sorted(set(self.competing_sources))) != self.competing_sources:
            raise ValueError("competing_sources must be unique and sorted")
        if self.accepted and self.selected_type is None:
            raise ValueError("Accepted proposal decisions require selected_type")

    def to_json(self, source_text: str) -> dict[str, Any]:
        start, end = self.span
        return {
            "span": [start, end],
            "text": source_text[start:end],
            "source": self.source,
            "candidate_types": [item.value for item in self.candidate_types],
            "accepted": self.accepted,
            "reason": self.reason,
            "selected_type": self.selected_type.value if self.selected_type else None,
            "competing_sources": list(self.competing_sources),
        }


@dataclass(frozen=True, slots=True)
class RuleNerTrace:
    """Complete proposal and decision lineage for one rule NER invocation."""

    proposals: tuple[EntityProposal, ...]
    decisions: tuple[ProposalDecision, ...]

    def to_json(self, source_text: str) -> dict[str, Any]:
        return {
            "proposals": [proposal.to_json(source_text) for proposal in self.proposals],
            "decisions": [decision.to_json(source_text) for decision in self.decisions],
        }
