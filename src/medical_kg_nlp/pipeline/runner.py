"""Stage orchestration for an already composed clinical NLP pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.pipeline.components import PipelineComponents
from medical_kg_nlp.pipeline.tracing import PipelineTrace
from medical_kg_nlp.preprocessing.section_splitter import split_sections
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument, Section, Sentence
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import CodeSystem
from medical_kg_nlp.schema.validator import PredictionValidator
from medical_kg_nlp.utils.text import text_window
from medical_kg_nlp.validation.profiles import (
    ValidationProfile,
    ValidationSeverity,
    apply_validation_profile,
)

__all__ = ["PipelineRunResult", "PipelineRunner"]


@dataclass(frozen=True)
class PipelineRunResult:
    """Prediction and machine-readable stage trace for one document."""

    prediction: ClinicalPrediction
    trace: PipelineTrace


class PipelineRunner:
    """Orchestrate stages using only injected components."""

    def __init__(self, components: PipelineComponents) -> None:
        self.components = components
        self.options = components.options

    def process_text(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, str] | None = None,
    ) -> ClinicalPrediction:
        return self.process_text_with_trace(document_id, text, metadata).prediction

    def process_text_with_trace(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, str] | None = None,
    ) -> PipelineRunResult:
        document = ClinicalDocument(document_id=document_id, text=text, metadata=metadata or {})
        return self.process_document_with_trace(document)

    def process_document(self, document: ClinicalDocument) -> ClinicalPrediction:
        return self.process_document_with_trace(document).prediction

    def process_document_with_trace(self, document: ClinicalDocument) -> PipelineRunResult:
        trace = PipelineTrace(document_id=document.document_id)
        with trace.stage("document_loader") as counters:
            loaded_document = self._load_document(document)
            counters["documents"] = 1
            counters["characters"] = len(loaded_document.text)
            counters["metadata_fields"] = len(loaded_document.metadata)

        with trace.stage("offset_preserving_preprocessing") as counters:
            mapped_text = self.components.normalization_contract.prepare(loaded_document.text)
            counters["original_characters"] = len(mapped_text.original)
            counters["normalized_characters"] = len(mapped_text.normalized)
            counters["offset_map_entries"] = len(mapped_text.normalized_to_original)
            # Keep this counter machine-readable for stage reports. The contract version is
            # recorded in code/config manifests rather than this integer-only trace structure.
            counters["diagnostic_only"] = int(
                self.components.normalization_contract.downstream_uses_source_text
            )

        with trace.stage("section_detection") as counters:
            sections = self._sections(loaded_document.text)
            counters["sections"] = len(sections)

        with trace.stage("sentence_splitting") as counters:
            sentences = self._sentences_from_sections(sections, loaded_document.text)
            counters["sentences"] = len(sentences)

        with trace.stage("entity_extraction") as counters:
            entities = self.components.entity_extractor.extract(loaded_document.text)
            counters["entities"] = len(entities)

        with trace.stage("context_assertion_classification") as counters:
            if self.options.enable_context:
                classifier = self.components.assertion_classifier
                if classifier is None:
                    raise RuntimeError("Assertion classifier component is unavailable.")
                for entity in entities:
                    sentence = self._find_sentence(entity, sentences)
                    features, evidence = classifier.classify_features_with_evidence(
                        entity,
                        sentence,
                    )
                    entity.assertion_features = features
                    entity.assertion = entity.assertion_features.primary()
                    entity.assertion_evidence = evidence
                    for item in evidence:
                        key = f"rule_{item.rule_id}"
                        counters[key] = counters.get(key, 0) + 1
                counters["classified_entities"] = len(entities)
                counters["matched_rule_events"] = sum(
                    count for key, count in counters.items() if key.startswith("rule_")
                )
            else:
                counters["skipped_entities"] = len(entities)

        contexts_by_entity: dict[str, str] = {}
        mentions_by_entity: dict[str, str] = {}
        generated_candidates: dict[str, list[Candidate]] = {}
        with trace.stage("candidate_generation") as counters:
            if self.options.enable_linking:
                retriever = self.components.candidate_retriever
                if retriever is None:
                    raise RuntimeError("Candidate retriever component is unavailable.")
                generated_total = 0
                entities_with_candidates = 0
                pinned_entities = 0
                for entity in entities:
                    if entity.code_system != CodeSystem.NONE:
                        pinned_entities += 1
                    context = text_window(
                        loaded_document.text, entity.span, radius=self.options.context_window
                    )
                    contexts_by_entity[entity.id] = context
                    mention = _linking_mention(loaded_document.text, entity)
                    mentions_by_entity[entity.id] = mention
                    candidates = retriever.retrieve(
                        entity,
                        context,
                        mention,
                    )
                    generated_candidates[entity.id] = candidates
                    generated_total += len(candidates)
                    entities_with_candidates += int(bool(candidates))
                    for candidate in candidates:
                        for source in candidate.sources:
                            key = f"source_{source}"
                            counters[key] = counters.get(key, 0) + 1
                counters["candidate_entities"] = len(generated_candidates)
                counters["pinned_entities"] = pinned_entities
                counters["entities_with_candidates"] = entities_with_candidates
                counters["generated_candidates"] = generated_total
                counters["candidate_sources"] = len(self.options.candidate_sources)
            else:
                counters["skipped_entities"] = len(entities)

        reranked_candidates: dict[str, list[Candidate]] = {}
        entities_by_id = {entity.id: entity for entity in entities}
        with trace.stage("candidate_reranking") as counters:
            if self.options.enable_linking:
                reranker = self.components.candidate_reranker
                for entity_id, candidates in generated_candidates.items():
                    if self.options.enable_candidate_reranking:
                        if reranker is None:
                            raise RuntimeError("Candidate reranker component is unavailable.")
                        ranked = reranker.rerank(
                            candidates,
                            contexts_by_entity.get(entity_id, ""),
                            mentions_by_entity.get(
                                entity_id,
                                _linking_mention(
                                    loaded_document.text, entities_by_id[entity_id]
                                ),
                            ),
                        )
                    else:
                        ranked = candidates
                    reranked_candidates[entity_id] = ranked
                counters["reranked_entities"] = len(reranked_candidates)
                counters["reranked_candidates"] = sum(
                    len(candidates) for candidates in reranked_candidates.values()
                )
                counters["skipped_reranking"] = (
                    0 if self.options.enable_candidate_reranking else len(reranked_candidates)
                )
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("graph_evidence_reranking") as counters:
            if self.options.enable_linking and self.options.enable_graph_evidence_reranking:
                document_reranker = self.components.document_candidate_reranker
                if document_reranker is None:
                    raise RuntimeError("Document candidate reranker component is unavailable.")
                reranked_candidates, graph_counters = document_reranker.rerank_document(
                    entities,
                    reranked_candidates,
                    sentences,
                    mentions_by_entity,
                )
                counters.update(graph_counters)
            elif self.options.enable_linking:
                counters["skipped_entities"] = len(reranked_candidates)
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("normalization_assignment") as counters:
            if self.options.enable_linking:
                assigner = self.components.candidate_assigner
                if assigner is None:
                    raise RuntimeError("Candidate assigner component is unavailable.")
                for entity in entities:
                    assigned_candidates = reranked_candidates.get(entity.id)
                    if assigned_candidates is None:
                        continue
                    assigner.assign(
                        entity,
                        assigned_candidates,
                        mention=_linking_mention(loaded_document.text, entity),
                    )
                counters["assigned_codes"] = sum(entity.code is not None for entity in entities)
                counters["unlinked_entities"] = sum(entity.code is None for entity in entities)
                counters["qualified_candidates"] = sum(
                    candidate.qualified
                    for entity in entities
                    for candidate in entity.candidates
                )
                counters["entities_with_qualified_candidates"] = sum(
                    any(candidate.qualified for candidate in entity.candidates)
                    for entity in entities
                )
                for entity in entities:
                    for schema_candidate in entity.candidates:
                        reason = schema_candidate.qualification_reason or "unspecified"
                        key = f"qualification_{reason}"
                        counters[key] = counters.get(key, 0) + 1
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("icd_rxnorm_umls_validation") as counters:
            if self.options.enable_entity_kg_validation:
                validator = self.components.knowledge_validator
                if validator is None:
                    raise RuntimeError("Knowledge validator component is unavailable.")
                entities, entity_issues = validator.validate_entities(entities)
                counters["issues"] = len(entity_issues)
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("relation_extraction") as counters:
            if self.options.enable_relations:
                extractor = self.components.relation_extractor
                if extractor is None:
                    raise RuntimeError("Relation extractor component is unavailable.")
                relations = extractor.extract(entities, sentences)
                counters["relations"] = len(relations)
            else:
                relations = []
                counters["skipped_entities"] = len(entities)

        with trace.stage("ontology_kg_consistency_check") as counters:
            if self.options.enable_relation_kg_validation:
                validator = self.components.knowledge_validator
                if validator is None:
                    raise RuntimeError("Knowledge validator component is unavailable.")
                relations, relation_issues = validator.validate_relations(entities, relations)
                counters["issues"] = len(relation_issues)
                counters["relations"] = len(relations)
            else:
                counters["relations"] = len(relations)

        with trace.stage("structured_json_output") as counters:
            prediction = ClinicalPrediction.from_text(
                document_id=loaded_document.document_id,
                text=loaded_document.text,
                entities=entities,
                relations=relations,
                pipeline_version=self.components.pipeline_version,
            )
            counters["entities"] = len(prediction.entities)
            counters["relations"] = len(prediction.relations)

        with trace.stage("prediction_validation") as counters:
            issues = PredictionValidator().validate_prediction(
                prediction,
                source_text=loaded_document.text,
            )
            profiled_issues = apply_validation_profile(
                issues,
                ValidationProfile.CORE,
                terminology_loaded=False,
            )
            errors = [
                item
                for item in profiled_issues
                if item.severity is ValidationSeverity.ERROR
            ]
            counters["issues"] = len(profiled_issues)
            counters["errors"] = len(errors)
            counters["warnings"] = len(profiled_issues) - len(errors)
            counters["validated_entities"] = len(prediction.entities)
            counters["validated_relations"] = len(prediction.relations)
            if errors:
                # INVARIANT: component adapters cannot bypass offset, type/code-system,
                # duplicate-ID, or relation safety by disabling optional KG stages.
                detail = "; ".join(
                    f"{item.issue.kind} at {item.issue.path}: {item.issue.message}"
                    for item in errors
                )
                raise ValueError(f"Core prediction validation failed: {detail}")
        return PipelineRunResult(prediction=prediction, trace=trace)

    @staticmethod
    def _load_document(document: ClinicalDocument) -> ClinicalDocument:
        return document

    @staticmethod
    def _sections(text: str) -> list[Section]:
        return split_sections(text)

    @staticmethod
    def _sentences_from_sections(sections: list[Section], source_text: str) -> list[Sentence]:
        sentences: list[Sentence] = []
        for section in sections:
            sentences.extend(
                split_sentences(
                    section.text, section_title=section.title, base_offset=section.span[0]
                )
            )
        return sentences or [Sentence(span=(0, len(source_text)), text=source_text)]

    @staticmethod
    def _find_sentence(entity: EntityAnnotation, sentences: list[Sentence]) -> Sentence | None:
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
