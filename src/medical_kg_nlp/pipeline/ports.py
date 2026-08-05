"""Stable contracts between pipeline orchestration and concrete implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from medical_kg_nlp.context.modifier_graph import AssertionDecision, ContextGraph
from medical_kg_nlp.kg.validator import ValidationIssue
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.batch import CandidateRerankRequest, CandidateRetrievalRequest
from medical_kg_nlp.schema.annotation import (
    AssertionEvidence,
    AssertionFeatures,
    EntityExtractionResult,
    EntityAnnotation,
    RelationAnnotation,
)
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.terminology.ports import TerminologyRepository

__all__ = [
    "AssertionClassifierPort",
    "BatchAssertionClassifierPort",
    "CandidateAssignerPort",
    "CandidateRerankerPort",
    "BatchCandidateRetrieverPort",
    "BatchCandidateRerankerPort",
    "CandidateRetrievalRequest",
    "CandidateRerankRequest",
    "DocumentCandidateRerankerPort",
    "CandidateRetrieverPort",
    "EntityExtractorPort",
    "EntityProposalExtractorPort",
    "KnowledgeValidatorPort",
    "RelationExtractorPort",
    "TerminologyRepository",
]


class EntityExtractorPort(Protocol):
    """Extract raw-text-backed entities from a clinical document."""

    def extract(self, source_text: str) -> list[EntityAnnotation]: ...


@runtime_checkable
class EntityProposalExtractorPort(Protocol):
    """Extract final entities while retaining unresolved type proposals."""

    def extract_with_proposals(self, source_text: str) -> EntityExtractionResult: ...


class AssertionClassifierPort(Protocol):
    """Classify assertion features while retaining rule/model evidence."""

    def classify_features_with_evidence(
        self,
        entity: EntityAnnotation,
        sentence: Sentence | None = None,
    ) -> tuple[AssertionFeatures, tuple[AssertionEvidence, ...]]: ...


@runtime_checkable
class BatchAssertionClassifierPort(Protocol):
    """Classify sentence targets together and expose modifier-target evidence."""

    def classify_batch_with_graph(
        self,
        entities: list[EntityAnnotation],
        sentence: Sentence,
    ) -> tuple[dict[str, AssertionDecision], ContextGraph]: ...


@runtime_checkable
class BatchCandidateRetrieverPort(Protocol):
    """Retrieve candidates for independent entities without sharing document state."""

    def retrieve_batch(
        self,
        requests: tuple[CandidateRetrievalRequest, ...],
    ) -> dict[str, list[Candidate]]: ...


@runtime_checkable
class BatchCandidateRerankerPort(Protocol):
    """Rerank independent candidate lists while preserving entity grouping."""

    def rerank_batch(
        self,
        requests: tuple[CandidateRerankRequest, ...],
    ) -> dict[str, list[Candidate]]: ...


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


class DocumentCandidateRerankerPort(Protocol):
    """Apply document-level evidence after mention-level candidate reranking."""

    def rerank_document(
        self,
        entities: list[EntityAnnotation],
        candidates_by_entity: dict[str, list[Candidate]],
        sentences: list[Sentence],
        mentions_by_entity: dict[str, str],
    ) -> tuple[dict[str, list[Candidate]], dict[str, int]]: ...


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
