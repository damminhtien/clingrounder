"""Adapters exposing deterministic implementations through pipeline ports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from medical_kg_nlp.context.assertion import AssertionClassifier
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.kg.validator import KGValidator, ValidationIssue
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.linker import EntityLinker
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.relations.rule_relations import RuleRelationExtractor
from medical_kg_nlp.schema.annotation import (
    AssertionEvidence,
    AssertionFeatures,
    EntityAnnotation,
    RelationAnnotation,
)
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.schema.types import CodeSystem, EntityType

__all__ = [
    "DictionaryCandidateAdapter",
    "InMemoryTerminologyRepository",
    "KGValidatorAdapter",
    "RuleAssertionClassifierAdapter",
    "RuleEntityExtractorAdapter",
    "RuleRelationExtractorAdapter",
]


@dataclass(frozen=True)
class RuleEntityExtractorAdapter:
    implementation: RuleBasedNER

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        entities = self.implementation.extract(source_text)
        # INVARIANT: adapters may enrich spans but never project them onto normalized text.
        for entity in entities:
            entity.validate_offsets(source_text)
        return entities


@dataclass(frozen=True)
class RuleAssertionClassifierAdapter:
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
    implementation: RuleRelationExtractor

    def extract(
        self,
        entities: list[EntityAnnotation],
        sentences: list[Sentence],
    ) -> list[RelationAnnotation]:
        return self.implementation.extract(entities, sentences)


@dataclass(frozen=True)
class KGValidatorAdapter:
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


@dataclass(frozen=True)
class InMemoryTerminologyRepository:
    """Port adapter for the recognition-sized in-memory dictionary."""

    store: DictionaryStore

    def get_by_concept_id(self, concept_id: str) -> ConceptEntry | None:
        return self.store.by_concept_id.get(concept_id)

    def get_by_code(self, code_system: CodeSystem, code: str) -> ConceptEntry | None:
        return self.store.by_code_system_code.get((code_system, code))

    def exact_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._filter(self.store.exact_lookup(mention), entity_type, code_systems, limit)

    def toneless_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._filter(self.store.toneless_lookup(mention), entity_type, code_systems, limit)

    def search(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        exact = self.exact_lookup(
            mention,
            entity_type=entity_type,
            code_systems=code_systems,
            limit=limit,
        )
        if len(exact) >= limit:
            return exact
        seen = {entry.concept_id for entry in exact}
        toneless = self.toneless_lookup(
            mention,
            entity_type=entity_type,
            code_systems=code_systems,
            limit=limit,
        )
        return [*exact, *(entry for entry in toneless if entry.concept_id not in seen)][:limit]

    @staticmethod
    def _filter(
        entries: list[ConceptEntry],
        entity_type: EntityType | None,
        code_systems: Sequence[CodeSystem] | None,
        limit: int,
    ) -> list[ConceptEntry]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        allowed_systems = set(code_systems) if code_systems is not None else None
        return [
            entry
            for entry in entries
            if (entity_type is None or entry.semantic_type == entity_type)
            and (allowed_systems is None or entry.code_system in allowed_systems)
        ][:limit]

