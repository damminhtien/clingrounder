from __future__ import annotations
from collections.abc import Mapping

from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.reranker import HeuristicReranker
from medical_kg_nlp.linking.structured_rxnorm import rxnorm_structure_conflict
from medical_kg_nlp.retrieval.candidate_generator import CandidateGenerator
from medical_kg_nlp.schema.annotation import CandidateConcept, EntityAnnotation
from medical_kg_nlp.schema.types import CodeSystem, EntityType


class EntityLinker:
    def __init__(
        self,
        generator: CandidateGenerator,
        *,
        assignment_threshold: float = 0.75,
        assignment_margin: float = 0.05,
        candidate_threshold: float | None = None,
        candidate_relative_margin: float | None = None,
        max_qualified_candidates: int = 5,
        candidate_thresholds_by_entity_type: Mapping[EntityType, float] | None = None,
        candidate_thresholds_by_source: Mapping[str, float] | None = None,
        emit_probabilities_by_source: Mapping[str, float] | None = None,
        enforce_rxnorm_structure: bool = True,
    ) -> None:
        effective_candidate_threshold = (
            assignment_threshold if candidate_threshold is None else candidate_threshold
        )
        effective_relative_margin = (
            assignment_margin
            if candidate_relative_margin is None
            else candidate_relative_margin
        )
        if not 0.0 <= effective_candidate_threshold <= 1.0:
            raise ValueError("candidate_threshold must be between 0 and 1")
        if not 0.0 <= effective_relative_margin <= 1.0:
            raise ValueError("candidate_relative_margin must be between 0 and 1")
        if not 1 <= max_qualified_candidates <= 5:
            raise ValueError("max_qualified_candidates must be between 1 and 5")
        type_thresholds = dict(candidate_thresholds_by_entity_type or {})
        source_thresholds = dict(candidate_thresholds_by_source or {})
        emit_probabilities = dict(emit_probabilities_by_source or {})
        if any(not 0.0 <= threshold <= 1.0 for threshold in type_thresholds.values()):
            raise ValueError("candidate type thresholds must be between 0 and 1")
        if any(not 0.0 <= threshold <= 1.0 for threshold in source_thresholds.values()):
            raise ValueError("candidate source thresholds must be between 0 and 1")
        if any(not 0.0 <= value <= 1.0 for value in emit_probabilities.values()):
            raise ValueError("candidate emit probabilities must be between 0 and 1")
        self.generator = generator
        self.reranker = HeuristicReranker(generator.store)
        self.assignment_threshold = assignment_threshold
        self.assignment_margin = assignment_margin
        self.candidate_threshold = effective_candidate_threshold
        self.candidate_relative_margin = effective_relative_margin
        self.max_qualified_candidates = max_qualified_candidates
        self.candidate_thresholds_by_entity_type = type_thresholds
        self.candidate_thresholds_by_source = source_thresholds
        self.emit_probabilities_by_source = emit_probabilities
        self.enforce_rxnorm_structure = enforce_rxnorm_structure

    def link_entity(self, entity: EntityAnnotation, context_window: str = "") -> EntityAnnotation:
        generated = self.generate_candidates(entity, context_window)
        candidates = self.rerank_candidates(generated, context_window, entity.text)
        return self.apply_candidates(entity, candidates, mention=entity.text)

    def generate_candidates(
        self,
        entity: EntityAnnotation,
        context_window: str = "",
        mention: str | None = None,
    ) -> list[Candidate]:
        full_mention = mention or entity.text
        candidates = self.generator.generate(full_mention, entity.type, context_window)
        if full_mention == entity.text or any(
            candidate.source == "btc_sample" for candidate in candidates
        ):
            return candidates
        candidates.extend(self.generator.generate(entity.text, entity.type, context_window))
        by_concept: dict[str, Candidate] = {}
        for candidate in candidates:
            previous = by_concept.get(candidate.concept_id)
            if previous is None or candidate.score > previous.score:
                by_concept[candidate.concept_id] = candidate
        return sorted(by_concept.values(), key=lambda item: item.score, reverse=True)[
            : self.generator.max_candidates
        ]

    def rerank_candidates(
        self,
        candidates: list[Candidate],
        context_window: str = "",
        mention: str = "",
    ) -> list[Candidate]:
        return self.reranker.rerank(candidates, context_window, mention)

    def apply_candidates(
        self,
        entity: EntityAnnotation,
        candidates: list[Candidate],
        *,
        mention: str | None = None,
    ) -> EntityAnnotation:
        entity.candidates = self._qualify_candidates(
            entity,
            candidates,
            mention=mention or entity.text,
        )
        if self._should_assign(candidates, entity.candidates):
            top = candidates[0]
            entity.code_system = top.code_system
            entity.code = top.code
            entity.confidence = max(entity.confidence, top.score)
        elif entity.code_system == CodeSystem.NONE:
            entity.code = None
            entity.confidence = max(entity.confidence, 0.5)
        return entity

    def _should_assign(
        self,
        candidates: list[Candidate],
        schema_candidates: list[CandidateConcept],
    ) -> bool:
        if not candidates or candidates[0].code is None:
            return False
        if not schema_candidates or not schema_candidates[0].qualified:
            return False
        top_score = candidates[0].score
        if top_score < self.assignment_threshold:
            return False
        if len(candidates) == 1:
            return True
        return top_score - candidates[1].score >= self.assignment_margin

    def _qualify_candidates(
        self,
        entity: EntityAnnotation,
        candidates: list[Candidate],
        *,
        mention: str,
    ) -> list[CandidateConcept]:
        if not candidates:
            return []
        top_score = candidates[0].score
        qualified_count = 0
        schema_candidates: list[CandidateConcept] = []
        for candidate in candidates:
            qualified, reason = self._qualification(
                candidate,
                entity_type=entity.type,
                mention=mention,
                top_score=top_score,
                qualified_count=qualified_count,
            )
            if qualified:
                qualified_count += 1
            schema_candidates.append(
                self._to_schema(
                    candidate,
                    qualified=qualified,
                    qualification_reason=reason,
                )
            )
        return schema_candidates

    def _qualification(
        self,
        candidate: Candidate,
        *,
        entity_type: EntityType,
        mention: str,
        top_score: float,
        qualified_count: int,
    ) -> tuple[bool, str]:
        if candidate.code is None:
            return False, "missing_code"
        if (
            self.enforce_rxnorm_structure
            and entity_type == EntityType.DRUG
            and candidate.code_system == CodeSystem.RXNORM
        ):
            entry = self.generator.store.by_concept_id.get(candidate.concept_id)
            if entry is not None:
                conflict = rxnorm_structure_conflict(mention, entry)
                if conflict is not None:
                    return False, conflict
        threshold = self.candidate_thresholds_by_source.get(
            candidate.source,
            self.candidate_thresholds_by_entity_type.get(
                entity_type,
                self.candidate_threshold,
            ),
        )
        if candidate.score < threshold:
            return False, "below_absolute_threshold"
        if top_score - candidate.score > self.candidate_relative_margin:
            return False, "outside_relative_margin"
        if qualified_count >= self.max_qualified_candidates:
            return False, "beyond_max_candidates"
        return True, "qualified"

    def _to_schema(
        self,
        candidate: Candidate,
        *,
        qualified: bool,
        qualification_reason: str,
    ) -> CandidateConcept:
        return CandidateConcept(
            concept_id=candidate.concept_id,
            code_system=candidate.code_system,
            code=candidate.code,
            name=candidate.canonical_name,
            retrieval_score=candidate.score,
            emit_probability=(
                self.emit_probabilities_by_source.get(
                    f"{candidate.code_system.value}:{candidate.source}",
                    self.emit_probabilities_by_source.get(candidate.source, 0.0),
                )
                if qualified
                else 0.0
            ),
            source=candidate.source,
            evidence_sources=candidate.sources,
            matched_alias=candidate.matched_alias or candidate.canonical_name,
            qualified=qualified,
            qualification_reason=qualification_reason,
        )
