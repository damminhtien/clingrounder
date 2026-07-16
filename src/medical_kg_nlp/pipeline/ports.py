"""Stable contracts between pipeline orchestration and concrete implementations."""

from __future__ import annotations

from typing import Protocol

from medical_kg_nlp.kg.validator import ValidationIssue
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.schema.annotation import (
    AssertionEvidence,
    AssertionFeatures,
    EntityAnnotation,
    RelationAnnotation,
)
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.terminology.ports import TerminologyRepository

__all__ = [
    "AssertionClassifierPort",
    "CandidateAssignerPort",
    "CandidateRerankerPort",
    "CandidateRetrieverPort",
    "EntityExtractorPort",
    "KnowledgeValidatorPort",
    "RelationExtractorPort",
    "TerminologyRepository",
]


class EntityExtractorPort(Protocol):
    """Extract raw-text-backed entities from a clinical document."""

    def extract(self, source_text: str) -> list[EntityAnnotation]: ...


class AssertionClassifierPort(Protocol):
    """Classify assertion features while retaining rule/model evidence."""

    def classify_features_with_evidence(
        self,
        entity: EntityAnnotation,
        sentence: Sentence | None = None,
    ) -> tuple[AssertionFeatures, tuple[AssertionEvidence, ...]]: ...


class CandidateRetrieverPort(Protocol):
    """Retrieve type-compatible normalization candidates."""

    def retrieve(
        self,
        entity: EntityAnnotation,
        context_window: str = "",
        mention: str | None = None,
    ) -> list[Candidate]: ...


class CandidateRerankerPort(Protocol):
    """Rerank a bounded candidate list for one mention."""

    def rerank(
        self,
        candidates: list[Candidate],
        context_window: str = "",
        mention: str = "",
    ) -> list[Candidate]: ...


class CandidateAssignerPort(Protocol):
    """Qualify candidates and assign the winning code to an entity."""

    def assign(
        self,
        entity: EntityAnnotation,
        candidates: list[Candidate],
        *,
        mention: str | None = None,
    ) -> EntityAnnotation: ...


class RelationExtractorPort(Protocol):
    """Extract typed relations between already extracted entities."""

    def extract(
        self,
        entities: list[EntityAnnotation],
        sentences: list[Sentence],
    ) -> list[RelationAnnotation]: ...


class KnowledgeValidatorPort(Protocol):
    """Apply entity/code and relation/KG constraints."""

    def validate_entities(
        self,
        entities: list[EntityAnnotation],
    ) -> tuple[list[EntityAnnotation], list[ValidationIssue]]: ...

    def validate_relations(
        self,
        entities: list[EntityAnnotation],
        relations: list[RelationAnnotation],
    ) -> tuple[list[RelationAnnotation], list[ValidationIssue]]: ...

