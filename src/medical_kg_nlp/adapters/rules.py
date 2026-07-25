"""Adapters exposing deterministic implementations through pipeline ports."""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.context.assertion import AssertionClassifier
from medical_kg_nlp.kg.validator import KGValidator, ValidationIssue
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.linker import EntityLinker
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.relations.rule_relations import RuleRelationExtractor
from medical_kg_nlp.schema.annotation import (
    AssertionEvidence,
    AssertionFeatures,
    EntityAnnotation,
    EntityExtractionResult,
    RelationAnnotation,
)
from medical_kg_nlp.schema.document import Sentence

__all__ = [
    "DictionaryCandidateAdapter",
    "KGValidatorAdapter",
    "RuleAssertionClassifierAdapter",
    "RuleEntityExtractorAdapter",
    "RuleRelationExtractorAdapter",
]


@dataclass(frozen=True)
class RuleEntityExtractorAdapter:
    """Expose deterministic dictionary/rule NER through the extraction port."""

    implementation: RuleBasedNER

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        entities = self.implementation.extract(source_text)
        # INVARIANT: adapters may enrich spans but never project them onto normalized text.
        for entity in entities:
            entity.validate_offsets(source_text)
        return entities

    def extract_with_proposals(self, source_text: str) -> EntityExtractionResult:
        """Retain unresolved dictionary type evidence for hybrid arbitration."""

        result = self.implementation.extract_with_proposals(source_text)
        for entity in result.entities:
            entity.validate_offsets(source_text)
        for proposal in result.ambiguous_proposals:
            proposal.validate_offsets(source_text)
        return result


@dataclass(frozen=True)
class RuleAssertionClassifierAdapter:
    """Expose cue-based assertion classification through the context port."""

    implementation: AssertionClassifier

    def classify_features_with_evidence(
        self,
        entity: EntityAnnotation,
        sentence: Sentence | None = None,
    ) -> tuple[AssertionFeatures, tuple[AssertionEvidence, ...]]:
        return self.implementation.classify_features_with_evidence(entity, sentence)


@dataclass(frozen=True)
class DictionaryCandidateAdapter:
    """Expose the current linker as retrieval, reranking, and assignment ports."""

    implementation: EntityLinker

    def retrieve(
        self,
        entity: EntityAnnotation,
        context_window: str = "",
        mention: str | None = None,
    ) -> list[Candidate]:
        return self.implementation.generate_candidates(entity, context_window, mention)

    def rerank(
        self,
        candidates: list[Candidate],
        context_window: str = "",
        mention: str = "",
    ) -> list[Candidate]:
        return self.implementation.rerank_candidates(candidates, context_window, mention)

    def assign(
        self,
        entity: EntityAnnotation,
        candidates: list[Candidate],
        *,
        mention: str | None = None,
    ) -> EntityAnnotation:
        return self.implementation.apply_candidates(entity, candidates, mention=mention)


@dataclass(frozen=True)
class RuleRelationExtractorAdapter:
    """Expose deterministic typed relation extraction through the relation port."""

    implementation: RuleRelationExtractor

    def extract(
        self,
        entities: list[EntityAnnotation],
        sentences: list[Sentence],
    ) -> list[RelationAnnotation]:
        return self.implementation.extract(entities, sentences)


@dataclass(frozen=True)
class KGValidatorAdapter:
    """Expose entity and relation constraints through one validator port."""

    implementation: KGValidator

    def validate_entities(
        self,
        entities: list[EntityAnnotation],
    ) -> tuple[list[EntityAnnotation], list[ValidationIssue]]:
        return self.implementation.validate_entities(entities)

    def validate_relations(
        self,
        entities: list[EntityAnnotation],
        relations: list[RelationAnnotation],
    ) -> tuple[list[RelationAnnotation], list[ValidationIssue]]:
        return self.implementation.validate_relations(entities, relations)
