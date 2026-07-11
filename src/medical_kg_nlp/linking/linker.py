from __future__ import annotations
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.reranker import HeuristicReranker
from medical_kg_nlp.retrieval.candidate_generator import CandidateGenerator
from medical_kg_nlp.schema.annotation import CandidateConcept, EntityAnnotation
from medical_kg_nlp.schema.types import CodeSystem


class EntityLinker:
    def __init__(
        self,
        generator: CandidateGenerator,
        *,
        assignment_threshold: float = 0.75,
        assignment_margin: float = 0.05,
    ) -> None:
        self.generator = generator
        self.reranker = HeuristicReranker(generator.store)
        self.assignment_threshold = assignment_threshold
        self.assignment_margin = assignment_margin

    def link_entity(self, entity: EntityAnnotation, context_window: str = "") -> EntityAnnotation:
        generated = self.generate_candidates(entity, context_window)
        candidates = self.rerank_candidates(generated, context_window, entity.text)
        return self.apply_candidates(entity, candidates)

    def generate_candidates(
        self, entity: EntityAnnotation, context_window: str = ""
    ) -> list[Candidate]:
        return self.generator.generate(entity.text, entity.type, context_window)

    def rerank_candidates(
        self,
        candidates: list[Candidate],
        context_window: str = "",
        mention: str = "",
    ) -> list[Candidate]:
        return self.reranker.rerank(candidates, context_window, mention)

    def apply_candidates(
        self, entity: EntityAnnotation, candidates: list[Candidate]
    ) -> EntityAnnotation:
        entity.candidates = [self._to_schema(candidate) for candidate in candidates]
        if self._should_assign(candidates):
            top = candidates[0]
            entity.code_system = top.code_system
            entity.code = top.code
            entity.confidence = max(entity.confidence, top.score)
        elif entity.code_system == CodeSystem.NONE:
            entity.code = None
            entity.confidence = max(entity.confidence, 0.5)
        return entity

    def _should_assign(self, candidates: list[Candidate]) -> bool:
        if not candidates or candidates[0].code is None:
            return False
        top_score = candidates[0].score
        if top_score < self.assignment_threshold:
            return False
        if len(candidates) == 1:
            return True
        return top_score - candidates[1].score >= self.assignment_margin

    @staticmethod
    def _to_schema(candidate: Candidate) -> CandidateConcept:
        return CandidateConcept(
            concept_id=candidate.concept_id,
            code_system=candidate.code_system,
            code=candidate.code,
            name=candidate.canonical_name,
            score=candidate.score,
            source="+".join(candidate.sources),
            matched_alias=candidate.matched_alias,
        )
