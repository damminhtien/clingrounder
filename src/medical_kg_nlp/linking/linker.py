from __future__ import annotations
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.reranker import HeuristicReranker
from medical_kg_nlp.retrieval.candidate_generator import CandidateGenerator
from medical_kg_nlp.schema.annotation import CandidateConcept, EntityAnnotation
from medical_kg_nlp.schema.types import CodeSystem


class EntityLinker:
    def __init__(self, generator: CandidateGenerator) -> None:
        self.generator = generator
        self.reranker = HeuristicReranker()

    def link_entity(self, entity: EntityAnnotation, context_window: str = "") -> EntityAnnotation:
        generated = self.generator.generate(entity.text, entity.type, context_window)
        candidates = self.reranker.rerank(generated, context_window)
        entity.candidates = [self._to_schema(candidate) for candidate in candidates]
        if candidates:
            top = candidates[0]
            entity.code_system = top.code_system
            entity.code = top.code
            entity.confidence = max(entity.confidence, top.score)
        elif entity.code_system == CodeSystem.NONE:
            entity.confidence = max(entity.confidence, 0.5)
        return entity

    @staticmethod
    def _to_schema(candidate: Candidate) -> CandidateConcept:
        return CandidateConcept(
            concept_id=candidate.concept_id,
            code_system=candidate.code_system,
            code=candidate.code,
            name=candidate.canonical_name,
            score=candidate.score,
            source=candidate.source,
            matched_alias=candidate.matched_alias,
        )

