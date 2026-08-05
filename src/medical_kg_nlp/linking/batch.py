"""Dependency-light request records for batched candidate operations."""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.schema.types import EntityType

__all__ = ["CandidateRetrievalRequest", "CandidateRerankRequest"]


@dataclass(frozen=True)
class CandidateRetrievalRequest:
    """One entity request passed to an optional batch retriever."""

    entity_id: str
    entity_type: EntityType
    mention: str
    context_window: str


@dataclass(frozen=True)
class CandidateRerankRequest:
    """One bounded candidate list passed to an optional batch reranker."""

    entity_id: str
    mention: str
    context_window: str
    candidates: tuple[Candidate, ...]
