"""Typed execution stages used by :mod:`medical_kg_nlp.pipeline.runner`.

Stages own transformations and validation for one concern.  They do not own expensive model or
repository lifecycles; those remain in ``PipelineRuntime``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from medical_kg_nlp.kg.validator import ValidationIssue
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.pipeline.ports import (
    AssertionClassifierPort,
    BatchAssertionClassifierPort,
    BatchCandidateRerankerPort,
    BatchCandidateRetrieverPort,
    CandidateAssignerPort,
    CandidateRerankerPort,
    CandidateRetrieverPort,
    DocumentCandidateRerankerPort,
    EntityExtractorPort,
    KnowledgeValidatorPort,
    RelationExtractorPort,
)
from medical_kg_nlp.preprocessing.normalizer import NormalizationContract
from medical_kg_nlp.preprocessing.section_rules import split_sections
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.schema.annotation import (
    AssertionEvidence,
    AssertionFeatures,
    EntityAnnotation,
    RelationAnnotation,
)
from medical_kg_nlp.schema.document import ClinicalDocument, Section, Sentence
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.validator import PredictionValidationIssue, PredictionValidator
from medical_kg_nlp.schema.types import CodeSystem
from medical_kg_nlp.utils.text import text_window
from medical_kg_nlp.linking.batch import (
    CandidateRerankRequest,
    CandidateRetrievalRequest,
)
from medical_kg_nlp.validation.profiles import (
    ProfiledValidationIssue,
    ValidationProfile,
    ValidationSeverity,
    apply_validation_profile,
)

__all__ = [
    "AssertionClassificationStage",
    "CandidateGenerationResult",
    "CandidateGenerationStage",
    "CandidateRerankingResult",
    "CandidateRerankingStage",
    "DocumentPreparationStage",
    "DocumentStructure",
    "LinkingContext",
    "LinkingStageResult",
    "PreparedDocument",
    "EntityExtractionStage",
    "EntityKnowledgeValidationResult",
    "EntityKnowledgeValidationStage",
    "GraphEvidenceRerankingStage",
    "NormalizationAssignmentStage",
    "PredictionValidationResult",
    "PredictionValidationStage",
    "RelationExtractionResult",
    "RelationExtractionStage",
]


@dataclass(frozen=True)
class DocumentStructure:
    """Source-coordinate sections and sentences for one immutable document."""

    sections: tuple[Section, ...]
    sentences: tuple[Sentence, ...]


@dataclass(frozen=True)
class PreparedDocument:
    """Prepared source document and its source-coordinate structure."""

    document: ClinicalDocument
    structure: DocumentStructure

    @property
    def sections(self) -> tuple[Section, ...]:
        return self.structure.sections

    @property
    def sentences(self) -> tuple[Sentence, ...]:
        return self.structure.sentences


@dataclass(frozen=True)
class LinkingContext:
    """Immutable mention/context projections shared by candidate stages."""

    mentions_by_entity: Mapping[str, str]
    contexts_by_entity: Mapping[str, str]


@dataclass(frozen=True)
class LinkingStageResult:
    """All linking intermediates aligned by entity ID and owned by one stage result."""

    context: LinkingContext
    generated_candidates: Mapping[str, list[Candidate]]
    reranked_candidates: Mapping[str, list[Candidate]]


class DocumentPreparationStage:
    """Prepare source structure without changing the document text or offsets."""

    def prepare(self, document: ClinicalDocument) -> PreparedDocument:
        """Create one typed preparation value for all downstream document stages."""

        return PreparedDocument(document=document, structure=self.structure(document.text))

    def diagnostics(
        self,
        source_text: str,
        contract: NormalizationContract,
    ) -> dict[str, int]:
        """Return lookup diagnostics while keeping normalized text out of later stages."""

        mapped_text = contract.prepare(source_text)
        return {
            "original_characters": len(mapped_text.original),
            "normalized_characters": len(mapped_text.normalized),
            "offset_map_entries": len(mapped_text.normalized_to_original),
            "source_coordinate_spans": 1,
            "normalized_text_used_downstream": 0,
        }

    def structure(self, source_text: str) -> DocumentStructure:
        """Detect sections and split sentences in source coordinates."""

        sections = tuple(split_sections(source_text))
        sentences: list[Sentence] = []
        for section in sections:
            sentences.extend(
                split_sentences(
                    section.text,
                    section_title=section.title,
                    base_offset=section.span[0],
                )
            )
        if not sentences:
            sentences.append(Sentence(span=(0, len(source_text)), text=source_text))
        return DocumentStructure(sections=sections, sentences=tuple(sentences))


@dataclass(frozen=True)
class EntityExtractionStage:
    """Run one injected entity extractor and enforce source-coordinate spans."""

    extractor: EntityExtractorPort

    def run(self, source_text: str) -> list[EntityAnnotation]:
        entities = self.extractor.extract(source_text)
        for entity in entities:
            # INVARIANT: a stage may enrich an entity but never move it off the raw source.
            entity.validate_offsets(source_text)
        return entities


@dataclass(frozen=True)
class CandidateGenerationResult:
    """Candidate retrieval output kept aligned by immutable entity IDs."""

    contexts_by_entity: dict[str, str]
    mentions_by_entity: dict[str, str]
    candidates_by_entity: dict[str, list[Candidate]]
    counters: dict[str, int]


@dataclass(frozen=True)
class CandidateGenerationStage:
    """Retrieve candidates without owning terminology or retriever lifecycles."""

    retriever: CandidateRetrieverPort | None
    context_window: int

    def run(
        self,
        source_text: str,
        entities: list[EntityAnnotation],
    ) -> CandidateGenerationResult:
        if self.retriever is None:
            raise RuntimeError("Candidate retriever component is unavailable.")

        contexts: dict[str, str] = {}
        mentions: dict[str, str] = {}
        requests: list[CandidateRetrievalRequest] = []
        counters: dict[str, int] = {}
        pinned_entities = 0
        for entity in entities:
            if entity.code_system != CodeSystem.NONE:
                pinned_entities += 1
            context = text_window(source_text, entity.span, radius=self.context_window)
            contexts[entity.id] = context
            mention = _linking_mention(source_text, entity)
            mentions[entity.id] = mention
            requests.append(
                CandidateRetrievalRequest(
                    entity_id=entity.id,
                    entity_type=entity.type,
                    mention=mention,
                    context_window=context,
                )
            )

        if isinstance(self.retriever, BatchCandidateRetrieverPort):
            retrieved = self.retriever.retrieve_batch(tuple(requests))
            _validate_batch_keys(
                tuple(request.entity_id for request in requests),
                retrieved,
                stage="candidate retrieval",
            )
        else:
            retrieved = {
                request.entity_id: self.retriever.retrieve(
                    entities[index], request.context_window, request.mention
                )
                for index, request in enumerate(requests)
            }

        candidates_by_entity: dict[str, list[Candidate]] = {}
        generated_total = 0
        entities_with_candidates = 0
        for request in requests:
            candidates = list(retrieved.get(request.entity_id, []))
            candidates_by_entity[request.entity_id] = candidates
            generated_total += len(candidates)
            entities_with_candidates += int(bool(candidates))
            bucket = min(len(candidates), 10)
            counters[f"candidate_count_{bucket}"] = (
                counters.get(f"candidate_count_{bucket}", 0) + 1
            )
            for candidate in candidates:
                for source in candidate.sources:
                    key = f"source_{source}"
                    counters[key] = counters.get(key, 0) + 1
        counters.update(
            {
                "candidate_entities": len(candidates_by_entity),
                "pinned_entities": pinned_entities,
                "entities_with_candidates": entities_with_candidates,
                "generated_candidates": generated_total,
            }
        )
        return CandidateGenerationResult(contexts, mentions, candidates_by_entity, counters)


@dataclass(frozen=True)
class CandidateRerankingResult:
    """Mention-level reranking output with candidate identity preserved."""

    candidates_by_entity: dict[str, list[Candidate]]
    counters: dict[str, int]


@dataclass(frozen=True)
class CandidateRerankingStage:
    """Rerank independent candidate lists using scalar or batch ports."""

    reranker: CandidateRerankerPort | None
    enabled: bool

    def run(
        self,
        entities: list[EntityAnnotation],
        candidates_by_entity: Mapping[str, list[Candidate]],
        contexts_by_entity: Mapping[str, str],
        mentions_by_entity: Mapping[str, str],
    ) -> CandidateRerankingResult:
        if self.enabled and self.reranker is None:
            raise RuntimeError("Candidate reranker component is unavailable.")
        entity_by_id = {entity.id: entity for entity in entities}
        requests = tuple(
            CandidateRerankRequest(
                entity_id=entity_id,
                mention=mentions_by_entity.get(entity_id, entity_by_id[entity_id].text),
                context_window=contexts_by_entity.get(entity_id, ""),
                candidates=tuple(candidates),
            )
            for entity_id, candidates in candidates_by_entity.items()
        )
        if self.enabled and isinstance(self.reranker, BatchCandidateRerankerPort):
            ranked = self.reranker.rerank_batch(requests)
            _validate_batch_keys(
                tuple(request.entity_id for request in requests),
                ranked,
                stage="candidate reranking",
            )
        else:
            ranked = {
                request.entity_id: (
                    self.reranker.rerank(
                        list(request.candidates), request.context_window, request.mention
                    )
                    if self.enabled and self.reranker is not None
                    else list(request.candidates)
                )
                for request in requests
            }
        output = {
            request.entity_id: list(ranked.get(request.entity_id, request.candidates))
            for request in requests
        }
        counters = {
            "reranked_entities": len(output),
            "reranked_candidates": sum(len(candidates) for candidates in output.values()),
            "skipped_reranking": 0 if self.enabled else len(output),
        }
        stats = getattr(self.reranker, "stats", None)
        if callable(stats):
            for name, value in stats().items():
                if isinstance(value, int):
                    counters[name] = value
        return CandidateRerankingResult(output, counters)


@dataclass(frozen=True)
class GraphEvidenceRerankingStage:
    """Apply optional document-level evidence after mention-level ranking."""

    reranker: DocumentCandidateRerankerPort | None

    def run(
        self,
        entities: list[EntityAnnotation],
        candidates_by_entity: Mapping[str, list[Candidate]],
        sentences: list[Sentence],
        mentions_by_entity: Mapping[str, str],
    ) -> tuple[dict[str, list[Candidate]], dict[str, int]]:
        if self.reranker is None:
            raise RuntimeError("Document candidate reranker component is unavailable.")
        return self.reranker.rerank_document(
            entities, candidates_by_entity, sentences, mentions_by_entity
        )


@dataclass(frozen=True)
class NormalizationAssignmentStage:
    """Assign qualified candidates without changing source spans."""

    assigner: CandidateAssignerPort | None

    def run(
        self,
        source_text: str,
        entities: list[EntityAnnotation],
        candidates_by_entity: Mapping[str, list[Candidate]],
    ) -> dict[str, int]:
        if self.assigner is None:
            raise RuntimeError("Candidate assigner component is unavailable.")
        counters: dict[str, int] = {}
        for entity in entities:
            candidates = candidates_by_entity.get(entity.id)
            if candidates is None:
                continue
            self.assigner.assign(entity, candidates, mention=_linking_mention(source_text, entity))
        counters["assigned_codes"] = sum(entity.code is not None for entity in entities)
        counters["unlinked_entities"] = sum(entity.code is None for entity in entities)
        counters["qualified_candidates"] = sum(
            candidate.qualified for entity in entities for candidate in entity.candidates
        )
        counters["entities_with_qualified_candidates"] = sum(
            any(candidate.qualified for candidate in entity.candidates) for entity in entities
        )
        for entity in entities:
            for candidate in entity.candidates:
                reason = candidate.qualification_reason or "unspecified"
                key = f"qualification_{reason}"
                counters[key] = counters.get(key, 0) + 1
        return counters


@dataclass(frozen=True)
class EntityKnowledgeValidationResult:
    """Entity validation output before prediction serialization."""

    entities: list[EntityAnnotation]
    issues: list[ValidationIssue]


@dataclass(frozen=True)
class EntityKnowledgeValidationStage:
    """Validate assigned entity codes against the active terminology release."""

    validator: KnowledgeValidatorPort | None

    def run(self, entities: list[EntityAnnotation]) -> EntityKnowledgeValidationResult:
        if self.validator is None:
            raise RuntimeError("Knowledge validator component is unavailable.")
        validated, issues = self.validator.validate_entities(entities)
        return EntityKnowledgeValidationResult(validated, issues)


@dataclass(frozen=True)
class RelationExtractionResult:
    """Relations emitted by the configured extractor for one document."""

    relations: list[RelationAnnotation]
    counters: dict[str, int]


@dataclass(frozen=True)
class RelationExtractionStage:
    """Extract typed relations; ontology validation remains a separate trace stage."""

    extractor: RelationExtractorPort | None

    def run(
        self,
        entities: list[EntityAnnotation],
        sentences: list[Sentence],
    ) -> RelationExtractionResult:
        if self.extractor is None:
            raise RuntimeError("Relation extractor component is unavailable.")
        relations = self.extractor.extract(entities, sentences)
        return RelationExtractionResult(relations, {"relations": len(relations)})


@dataclass(frozen=True)
class AssertionClassificationStage:
    """Apply scalar or batch assertion ports to entities inside sentence scope."""

    classifier: AssertionClassifierPort | None

    def run(
        self,
        entities: list[EntityAnnotation],
        sentences: tuple[Sentence, ...],
        counters: dict[str, int],
    ) -> None:
        if self.classifier is None:
            raise RuntimeError("Assertion classifier component is unavailable.")
        classified_ids: set[str] = set()
        if isinstance(self.classifier, BatchAssertionClassifierPort):
            for sentence in sentences:
                sentence_entities = [
                    entity
                    for entity in entities
                    if sentence.span[0] <= entity.span[0] < entity.span[1] <= sentence.span[1]
                ]
                if not sentence_entities:
                    continue
                decisions, graph = self.classifier.classify_batch_with_graph(
                    sentence_entities,
                    sentence,
                )
                for entity in sentence_entities:
                    features, evidence = decisions[entity.id]
                    _apply_assertion_decision(entity, features, evidence, counters)
                    classified_ids.add(entity.id)
                counters["context_graph_targets"] = (
                    counters.get("context_graph_targets", 0) + len(graph.targets)
                )
                counters["context_graph_modifiers"] = (
                    counters.get("context_graph_modifiers", 0) + len(graph.modifiers)
                )
                counters["context_graph_edges"] = (
                    counters.get("context_graph_edges", 0) + len(graph.edges)
                )
        for entity in entities:
            if entity.id in classified_ids:
                continue
            features, evidence = self.classifier.classify_features_with_evidence(
                entity,
                _find_sentence(entity, sentences),
            )
            _apply_assertion_decision(entity, features, evidence, counters)
        counters["classified_entities"] = len(entities)
        counters["matched_rule_events"] = sum(
            count for key, count in counters.items() if key.startswith("rule_")
        )


@dataclass(frozen=True)
class PredictionValidationResult:
    """Profiled validation output consumed by the runner's trace stage."""

    issues: tuple[ProfiledValidationIssue, ...]
    errors: tuple[ProfiledValidationIssue, ...]


@dataclass(frozen=True)
class PredictionValidationStage:
    """Validate a published prediction against the active terminology release."""

    validator: PredictionValidator
    profile: ValidationProfile = ValidationProfile.CORE

    def run(
        self,
        prediction: ClinicalPrediction,
        source_text: str,
    ) -> PredictionValidationResult:
        issues: list[PredictionValidationIssue] = self.validator.validate_prediction(
            prediction,
            source_text=source_text,
        )
        profiled = tuple(apply_validation_profile(issues, self.profile))
        errors = tuple(
            issue for issue in profiled if issue.severity is ValidationSeverity.ERROR
        )
        return PredictionValidationResult(issues=profiled, errors=errors)


def _apply_assertion_decision(
    entity: EntityAnnotation,
    features: AssertionFeatures,
    evidence: tuple[AssertionEvidence, ...],
    counters: dict[str, int],
) -> None:
    entity.assertion_features = features
    entity.assertion = features.primary()
    entity.assertion_evidence = evidence
    for item in evidence:
        key = f"rule_{item.rule_id}"
        counters[key] = counters.get(key, 0) + 1


def _find_sentence(
    entity: EntityAnnotation,
    sentences: tuple[Sentence, ...],
) -> Sentence | None:
    for sentence in sentences:
        if sentence.span[0] <= entity.span[0] and entity.span[1] <= sentence.span[1]:
            return sentence
    return None


def _linking_mention(source_text: str, entity: EntityAnnotation) -> str:
    medication = entity.medication_mention
    if medication is None:
        return entity.text
    start, end = medication.full_span
    return source_text[start:end]


def _validate_batch_keys(
    expected_ids: tuple[str, ...],
    results: dict[str, list[Candidate]],
    *,
    stage: str,
) -> None:
    """Reject incomplete batch responses before they can alter entity alignment."""

    expected = set(expected_ids)
    actual = set(results)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{stage} batch returned inconsistent entity IDs: "
            f"missing={missing}, unexpected={unexpected}"
        )
