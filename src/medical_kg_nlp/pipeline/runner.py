from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medical_kg_nlp.context.assertion import AssertionClassifier
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.kg.validator import KGValidator
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.linker import EntityLinker
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.pipeline.tracing import PipelineTrace
from medical_kg_nlp.preprocessing.normalizer import (
    DEFAULT_NORMALIZATION_CONTRACT,
    NormalizationContract,
)
from medical_kg_nlp.preprocessing.section_splitter import split_sections
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.relations.rule_relations import RuleRelationExtractor
from medical_kg_nlp.retrieval.candidate_generator import CandidateGenerator
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument, Section, Sentence
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import text_window


@dataclass(frozen=True)
class PipelineRunResult:
    prediction: ClinicalPrediction
    trace: PipelineTrace


class PipelineRunner:
    def __init__(
        self,
        dictionary_path: str | Path = "data/dictionaries/seed_concepts.jsonl",
        abbreviation_path: str | Path = "data/dictionaries/abbreviations.jsonl",
        alias_overlay_path: str | Path | None = "data/dictionaries/vietnamese_medical_alias.jsonl",
        recognition_dictionary_path: str | Path | None = None,
        normalization_dictionary_path: str | Path | None = None,
        pipeline_version: str = "0.1.0",
        options: PipelineOptions | None = None,
        normalization_contract: NormalizationContract = DEFAULT_NORMALIZATION_CONTRACT,
    ) -> None:
        self.options = options or PipelineOptions()
        self.normalization_contract = normalization_contract
        # Load raw entries first and build only the two stores the runtime actually keeps. Building
        # a temporary indexed store per source duplicates normalization work and becomes expensive
        # for complete RxNorm releases.
        recognition_entries = DictionaryStore.load_entries_jsonl(
            dictionary_path, alias_overlay_path=alias_overlay_path
        )
        if recognition_dictionary_path is not None:
            recognition_entries.extend(
                DictionaryStore.load_entries_jsonl(recognition_dictionary_path)
            )
        self.store = DictionaryStore(_deduplicate_entries(recognition_entries))
        self.normalization_store = (
            DictionaryStore(
                _deduplicate_entries(
                    [
                        *self.store.entries,
                        *DictionaryStore.load_entries_jsonl(normalization_dictionary_path),
                    ]
                )
            )
            if normalization_dictionary_path is not None
            else self.store
        )
        self.ner = RuleBasedNER(
            self.store,
            emit_probabilities_by_source=dict(
                self.options.link_emit_probabilities_by_source
            ),
        )
        self.linker = (
            EntityLinker(
                CandidateGenerator(
                    self.normalization_store,
                    abbreviation_path=abbreviation_path,
                    max_candidates=self.options.max_candidates,
                    retrieval_sources=self.options.candidate_sources,
                ),
                assignment_threshold=self.options.link_assignment_threshold,
                assignment_margin=self.options.link_assignment_margin,
                candidate_threshold=self.options.link_candidate_threshold,
                candidate_relative_margin=self.options.link_candidate_relative_margin,
                max_qualified_candidates=self.options.link_max_qualified_candidates,
                candidate_thresholds_by_entity_type={
                    EntityType(entity_type): threshold
                    for entity_type, threshold in self.options.link_candidate_thresholds_by_type
                },
                candidate_thresholds_by_source=dict(
                    self.options.link_candidate_thresholds_by_source
                ),
                emit_probabilities_by_source=dict(
                    self.options.link_emit_probabilities_by_source
                ),
                enforce_rxnorm_structure=self.options.link_enforce_rxnorm_structure,
            )
            if self.options.enable_linking
            else None
        )
        self.assertion = AssertionClassifier()
        self.relations = RuleRelationExtractor()
        self.validator = KGValidator(
            self.normalization_store
            if self.options.enable_entity_kg_validation
            or self.options.enable_relation_kg_validation
            else None
        )
        self.pipeline_version = pipeline_version

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
            mapped_text = self.normalization_contract.prepare(loaded_document.text)
            counters["original_characters"] = len(mapped_text.original)
            counters["normalized_characters"] = len(mapped_text.normalized)
            counters["offset_map_entries"] = len(mapped_text.normalized_to_original)
            # Keep this counter machine-readable for stage reports. The contract version is
            # recorded in code/config manifests rather than this integer-only trace structure.
            counters["diagnostic_only"] = int(
                self.normalization_contract.downstream_uses_source_text
            )

        with trace.stage("section_detection") as counters:
            sections = self._sections(loaded_document.text)
            counters["sections"] = len(sections)

        with trace.stage("sentence_splitting") as counters:
            sentences = self._sentences_from_sections(sections, loaded_document.text)
            counters["sentences"] = len(sentences)

        with trace.stage("entity_extraction") as counters:
            entities = self.ner.extract(loaded_document.text)
            counters["entities"] = len(entities)

        with trace.stage("context_assertion_classification") as counters:
            if self.options.enable_context:
                for entity in entities:
                    sentence = self._find_sentence(entity, sentences)
                    features, evidence = self.assertion.classify_features_with_evidence(
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
        generated_candidates: dict[str, list[Candidate]] = {}
        with trace.stage("candidate_generation") as counters:
            if self.options.enable_linking:
                linker = self._require_linker()
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
                    candidates = linker.generate_candidates(
                        entity,
                        context,
                        mention=_linking_mention(loaded_document.text, entity),
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
                linker = self._require_linker()
                for entity_id, candidates in generated_candidates.items():
                    if self.options.enable_candidate_reranking:
                        ranked = linker.rerank_candidates(
                            candidates,
                            contexts_by_entity.get(entity_id, ""),
                            _linking_mention(loaded_document.text, entities_by_id[entity_id]),
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

        with trace.stage("normalization_assignment") as counters:
            if self.options.enable_linking:
                linker = self._require_linker()
                for entity in entities:
                    assigned_candidates = reranked_candidates.get(entity.id)
                    if assigned_candidates is None:
                        continue
                    linker.apply_candidates(
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
                entities, entity_issues = self.validator.validate_entities(entities)
                counters["issues"] = len(entity_issues)
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("relation_extraction") as counters:
            if self.options.enable_relations:
                relations = self.relations.extract(entities, sentences)
                counters["relations"] = len(relations)
            else:
                relations = []
                counters["skipped_entities"] = len(entities)

        with trace.stage("ontology_kg_consistency_check") as counters:
            if self.options.enable_relation_kg_validation:
                relations, relation_issues = self.validator.validate_relations(entities, relations)
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
                pipeline_version=self.pipeline_version,
            )
            counters["entities"] = len(prediction.entities)
            counters["relations"] = len(prediction.relations)

        with trace.stage("prediction_validation") as counters:
            prediction.validate(loaded_document.text)
            counters["validated_entities"] = len(prediction.entities)
            counters["validated_relations"] = len(prediction.relations)
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

    def _require_linker(self) -> EntityLinker:
        if self.linker is None:
            raise RuntimeError("Linker is unavailable when enable_linking is false.")
        return self.linker


def _linking_mention(source_text: str, entity: EntityAnnotation) -> str:
    medication = entity.medication_mention
    if medication is None:
        return entity.text
    start, end = medication.full_span
    return source_text[start:end]


def _deduplicate_entries(entries: list[ConceptEntry]) -> list[ConceptEntry]:
    by_concept_id = {entry.concept_id: entry for entry in entries}
    return list(by_concept_id.values())
