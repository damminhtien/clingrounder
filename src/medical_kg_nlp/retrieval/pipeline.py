"""Storage-neutral candidate retrieval, fusion, and type constraints."""

from __future__ import annotations

from collections import defaultdict

from medical_kg_nlp.linking.candidate import Candidate, CandidateEvidence
from medical_kg_nlp.retrieval.adapters import MentionRetrieverAdapter
from medical_kg_nlp.retrieval.constraints import ALLOWED_CODE_SYSTEMS
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository

__all__ = ["ALLOWED_CODE_SYSTEMS", "RetrievalPipeline"]

SOURCE_WEIGHTS = {
    "exact": 1.0,
    "toneless": 0.92,
    "abbreviation": 0.92,
    "fuzzy": 0.72,
    "char_ngram": 0.62,
    "bm25": 0.50,
}
_RRF_K = 60
_PRIMARY_SCORE_WEIGHT = 0.85


class RetrievalPipeline:
    """Compose independently replaceable retrievers over one repository contract."""

    def __init__(
        self,
        repository: TerminologyRepository,
        retrievers: tuple[MentionRetrieverAdapter, ...],
        *,
        max_candidates: int = 20,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        self.repository = repository
        self.retrievers = retrievers
        self.max_candidates = max_candidates
        self.retrieval_sources = tuple(
            retriever.source for retriever in retrievers if retriever.source != "reviewed_memory"
        )

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str = "",
    ) -> list[Candidate]:
        """Retrieve, constrain, and deterministically fuse candidates for a mention."""

        candidates: list[Candidate] = []
        for retriever in self.retrievers:
            retrieved = retriever.retrieve(
                mention,
                entity_type,
                context_window,
                self.max_candidates,
            )
            constrained = [
                candidate
                for candidate in retrieved
                if self._allowed(candidate, entity_type)
            ]
            if constrained and retriever.terminal_on_match:
                return constrained[: self.max_candidates]
            candidates.extend(constrained)
            if retriever.unique_output_short_circuit:
                unique = self._unique_exact_output(constrained, entity_type)
                if unique is not None:
                    return [unique]
        merged = self._merge(candidates)
        return sorted(merged, key=lambda candidate: candidate.score, reverse=True)[
            : self.max_candidates
        ]

    @staticmethod
    def _allowed(candidate: Candidate, entity_type: EntityType) -> bool:
        allowed = ALLOWED_CODE_SYSTEMS.get(entity_type)
        if not allowed:
            return candidate.semantic_type == entity_type
        return candidate.semantic_type == entity_type and candidate.code_system in allowed

    def _unique_exact_output(
        self,
        candidates: list[Candidate],
        entity_type: EntityType,
    ) -> Candidate | None:
        exact = [
            candidate
            for candidate in candidates
            if candidate.source == "exact" and self._allowed(candidate, entity_type)
        ]
        if not exact or any(candidate.code is None for candidate in exact):
            return None
        if len({self._output_key(candidate) for candidate in exact}) != 1:
            return None
        merged = self._merge(exact)
        if len(merged) != 1:
            return None
        return self._replace(merged[0], score=max(candidate.score for candidate in exact))

    def _merge(self, candidates: list[Candidate]) -> list[Candidate]:
        by_source: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            by_source[candidate.source].append(candidate)

        evidence_by_code: dict[
            tuple[str, str], dict[str, tuple[CandidateEvidence, Candidate]]
        ] = defaultdict(dict)
        for source, source_candidates in sorted(by_source.items()):
            ranked = sorted(
                source_candidates,
                key=lambda item: (
                    -item.score,
                    item.code_system.value,
                    item.code or "",
                    item.concept_id,
                ),
            )
            for rank, candidate in enumerate(ranked, start=1):
                key = self._output_key(candidate)
                evidence = CandidateEvidence(
                    source=source,
                    score=candidate.score,
                    rank=rank,
                    concept_id=candidate.concept_id,
                    matched_alias=candidate.matched_alias,
                )
                current = evidence_by_code[key].get(source)
                if current is None or self._evidence_order(evidence) > self._evidence_order(
                    current[0]
                ):
                    evidence_by_code[key][source] = (evidence, candidate)

        merged: list[Candidate] = []
        rrf_normalizer = sum(weight / (_RRF_K + 1) for weight in SOURCE_WEIGHTS.values())
        for source_evidence in evidence_by_code.values():
            ordered = sorted(
                source_evidence.values(),
                key=lambda item: (
                    -(SOURCE_WEIGHTS.get(item[0].source, 0.0) * item[0].score),
                    item[0].rank,
                    item[0].source,
                    item[0].concept_id,
                ),
            )
            primary_evidence, primary = ordered[0]
            merged_evidence = tuple(item[0] for item in ordered)
            primary_score = max(
                SOURCE_WEIGHTS.get(item.source, 0.0) * item.score
                for item in merged_evidence
            )
            rrf_score = sum(
                SOURCE_WEIGHTS.get(item.source, 0.0) * item.score / (_RRF_K + item.rank)
                for item in merged_evidence
            )
            normalized_rrf = rrf_score / rrf_normalizer if rrf_normalizer else 0.0
            fused_score = min(
                1.0,
                _PRIMARY_SCORE_WEIGHT * primary_score
                + (1.0 - _PRIMARY_SCORE_WEIGHT) * normalized_rrf,
            )
            merged.append(
                self._replace(
                    primary,
                    score=fused_score,
                    source=primary_evidence.source,
                    matched_alias=primary_evidence.matched_alias,
                    evidence=merged_evidence,
                )
            )
        return merged

    @staticmethod
    def _output_key(candidate: Candidate) -> tuple[str, str]:
        if candidate.code:
            return candidate.code_system.value, candidate.code
        return "concept", candidate.concept_id

    @staticmethod
    def _evidence_order(evidence: CandidateEvidence) -> tuple[float, int, str]:
        return evidence.score, -evidence.rank, evidence.concept_id

    @staticmethod
    def _replace(
        candidate: Candidate,
        *,
        score: float | None = None,
        source: str | None = None,
        matched_alias: str | None = None,
        evidence: tuple[CandidateEvidence, ...] | None = None,
    ) -> Candidate:
        return Candidate(
            concept_id=candidate.concept_id,
            code=candidate.code,
            code_system=candidate.code_system,
            canonical_name=candidate.canonical_name,
            semantic_type=candidate.semantic_type,
            score=candidate.score if score is None else score,
            source=candidate.source if source is None else source,
            matched_alias=(
                candidate.matched_alias if matched_alias is None else matched_alias
            ),
            evidence=candidate.evidence if evidence is None else evidence,
        )
