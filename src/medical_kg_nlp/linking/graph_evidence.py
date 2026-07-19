"""Bounded graph-evidence features for candidate reranking.

The graph never creates a candidate.  It can only reorder candidates already
accepted by a type-constrained terminology retriever, using linked concepts from
the surrounding clinical context as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import log1p
from typing import TYPE_CHECKING

from medical_kg_nlp.kg.knowledge_schema import KnowledgeNeighbor
from medical_kg_nlp.kg.ports import KnowledgeGraphRepositoryPort
from medical_kg_nlp.linking.candidate import Candidate, CandidateEvidence
from medical_kg_nlp.schema.types import CodeSystem

if TYPE_CHECKING:
    from medical_kg_nlp.pipeline.ports import CandidateRerankerPort

__all__ = [
    "GraphContextConcept",
    "GraphEvidenceMatch",
    "GraphEvidenceReranker",
]


@dataclass(frozen=True)
class GraphContextConcept:
    """One trusted linked concept that may explain a target candidate."""

    code_system: CodeSystem
    code: str

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("Graph context code must be non-empty")
        if self.code_system == CodeSystem.NONE:
            raise ValueError("Graph context code system cannot be NONE")


@dataclass(frozen=True)
class GraphEvidenceMatch:
    """The strongest graph edge found between a candidate and one context concept."""

    context: GraphContextConcept
    relation_type: str
    support_count: int
    document_count: int
    confidence: float
    strength: float


class GraphEvidenceReranker:
    """Add a bounded graph feature after an optional lexical/model reranker.

    `context_concepts` should come from a first-pass linker or gold annotations in
    an explicitly labelled upper-bound benchmark.  Empty context preserves the
    base ranking exactly.
    """

    def __init__(
        self,
        repository: KnowledgeGraphRepositoryPort,
        *,
        base_reranker: CandidateRerankerPort | None = None,
        relation_types: tuple[str, ...] = ("CO_OCCURS_WITH",),
        min_support: int = 2,
        max_bonus: float = 0.04,
        support_saturation: int = 10,
        neighbor_limit: int = 500,
    ) -> None:
        if not relation_types or any(not value.strip() for value in relation_types):
            raise ValueError("Graph relation types must be non-empty")
        if min_support < 1 or support_saturation < 1 or neighbor_limit < 1:
            raise ValueError("Graph support and neighbor limits must be positive")
        if not 0.0 <= max_bonus <= 1.0:
            raise ValueError("Graph max_bonus must be in [0, 1]")
        self.repository = repository
        self.base_reranker = base_reranker
        self.relation_types = relation_types
        self.min_support = min_support
        self.max_bonus = max_bonus
        self.support_saturation = support_saturation
        self.neighbor_limit = neighbor_limit
        self._node_cache: dict[tuple[str, str], str | None] = {}
        self._neighbor_cache: dict[str, dict[str, KnowledgeNeighbor]] = {}

    def rerank(
        self,
        candidates: list[Candidate],
        context_window: str = "",
        mention: str = "",
        *,
        context_concepts: tuple[GraphContextConcept, ...] = (),
    ) -> list[Candidate]:
        """Rerank existing candidates and preserve deterministic tie-breaking."""

        base = (
            list(candidates)
            if self.base_reranker is None
            else self.base_reranker.rerank(
                candidates,
                context_window=context_window,
                mention=mention,
            )
        )
        if not context_concepts or self.max_bonus == 0.0:
            return base
        reranked = [
            self._score_candidate(candidate, context_concepts)
            for candidate in base
        ]
        return sorted(
            reranked,
            key=lambda candidate: (
                -candidate.score,
                candidate.code_system.value,
                candidate.code or "",
                candidate.concept_id,
            ),
        )

    def matches(
        self,
        candidate: Candidate,
        context_concepts: tuple[GraphContextConcept, ...],
    ) -> tuple[GraphEvidenceMatch, ...]:
        """Return auditable graph features without mutating a candidate."""

        candidate_node_id = self._node_id(candidate.code_system, candidate.code)
        if candidate_node_id is None:
            return ()
        output: list[GraphEvidenceMatch] = []
        seen_contexts: set[tuple[str, str]] = set()
        for context in context_concepts:
            context_key = (context.code_system.value, context.code)
            if context_key in seen_contexts:
                continue
            seen_contexts.add(context_key)
            context_node_id = self._node_id(context.code_system, context.code)
            if context_node_id is None or context_node_id == candidate_node_id:
                continue
            neighbor = self._neighbors(context_node_id).get(candidate_node_id)
            if neighbor is None:
                continue
            edge = neighbor.edge
            support = edge.document_count or edge.support_count
            strength = min(
                1.0,
                log1p(support) / log1p(self.support_saturation),
            ) * edge.confidence_mean
            output.append(
                GraphEvidenceMatch(
                    context=context,
                    relation_type=edge.relation_type,
                    support_count=edge.support_count,
                    document_count=edge.document_count,
                    confidence=edge.confidence_mean,
                    strength=strength,
                )
            )
        return tuple(
            sorted(
                output,
                key=lambda item: (
                    -item.strength,
                    item.context.code_system.value,
                    item.context.code,
                ),
            )
        )

    def _score_candidate(
        self,
        candidate: Candidate,
        context_concepts: tuple[GraphContextConcept, ...],
    ) -> Candidate:
        matches = self.matches(candidate, context_concepts)
        if not matches:
            return candidate
        # SCALING: independent context edges saturate instead of adding without bound.
        combined_strength = 1.0
        for match in matches:
            combined_strength *= 1.0 - match.strength
        combined_strength = 1.0 - combined_strength
        bonus = self.max_bonus * combined_strength
        evidence = candidate.evidence or (
            CandidateEvidence(
                source=candidate.source,
                score=candidate.score,
                rank=1,
                concept_id=candidate.concept_id,
                matched_alias=candidate.matched_alias,
            ),
        )
        graph_evidence = tuple(
            CandidateEvidence(
                source="graph_reranker",
                score=match.strength,
                rank=rank,
                concept_id=(
                    f"{match.context.code_system.value}:{match.context.code}"
                ),
                matched_alias=match.relation_type,
            )
            for rank, match in enumerate(matches, start=1)
        )
        # INVARIANT: graph evidence reorders dictionary candidates but never changes identity.
        return replace(
            candidate,
            score=min(1.0, candidate.score + bonus),
            evidence=(*evidence, *graph_evidence),
        )

    def _node_id(self, code_system: CodeSystem, code: str | None) -> str | None:
        if code is None or code_system == CodeSystem.NONE:
            return None
        key = (code_system.value, code)
        if key not in self._node_cache:
            node = self.repository.get_by_code(*key)
            self._node_cache[key] = None if node is None else node.node_id
        return self._node_cache[key]

    def _neighbors(self, node_id: str) -> dict[str, KnowledgeNeighbor]:
        cached = self._neighbor_cache.get(node_id)
        if cached is not None:
            return cached
        # SCALING: cache bounded index reads per context node for document-level reranking.
        neighbors = self.repository.neighbors(
            node_id,
            direction="both",
            relation_types=self.relation_types,
            min_support=self.min_support,
            limit=self.neighbor_limit,
        )
        by_node: dict[str, KnowledgeNeighbor] = {}
        for neighbor in neighbors:
            current = by_node.get(neighbor.node.node_id)
            if current is None or _neighbor_order(neighbor) > _neighbor_order(current):
                by_node[neighbor.node.node_id] = neighbor
        self._neighbor_cache[node_id] = by_node
        return by_node


def _neighbor_order(neighbor: KnowledgeNeighbor) -> tuple[int, int, float, str]:
    edge = neighbor.edge
    return (
        edge.document_count,
        edge.support_count,
        edge.confidence_mean,
        edge.edge_id,
    )
