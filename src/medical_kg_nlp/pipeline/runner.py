from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medical_kg_nlp.context.assertion import AssertionClassifier
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.kg.validator import KGValidator
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.linker import EntityLinker
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.pipeline.tracing import PipelineTrace
from medical_kg_nlp.preprocessing.offset_mapping import collapse_whitespace_preserve_offsets
from medical_kg_nlp.preprocessing.section_splitter import split_sections
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.relations.rule_relations import RuleRelationExtractor
from medical_kg_nlp.retrieval.candidate_generator import CandidateGenerator
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument, Section, Sentence
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import CodeSystem
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
        pipeline_version: str = "0.1.0",
        options: PipelineOptions | None = None,
    ) -> None:
        self.options = options or PipelineOptions()
        self.store = DictionaryStore.from_jsonl(dictionary_path)
        self.ner = RuleBasedNER(self.store)
        self.linker = EntityLinker(
            CandidateGenerator(
                self.store,
                abbreviation_path=abbreviation_path,
                max_candidates=self.options.max_candidates,
                retrieval_sources=self.options.candidate_sources,
            )
        )
        self.assertion = AssertionClassifier()
        self.relations = RuleRelationExtractor()
        self.validator = KGValidator()
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
            mapped_text = collapse_whitespace_preserve_offsets(loaded_document.text)
            counters["original_characters"] = len(mapped_text.original)
            counters["normalized_characters"] = len(mapped_text.normalized)
            counters["offset_map_entries"] = len(mapped_text.normalized_to_original)
            # Downstream stages still consume source text until normalized-span NER is wired end to end.
            counters["diagnostic_only"] = 1

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
                    entity.assertion = self.assertion.classify(entity, sentence)
                counters["classified_entities"] = len(entities)
            else:
                counters["skipped_entities"] = len(entities)

        contexts_by_entity: dict[str, str] = {}
        generated_candidates: dict[str, list[Candidate]] = {}
        with trace.stage("candidate_generation") as counters:
            if self.options.enable_linking:
                generated_total = 0
                entities_with_candidates = 0
                for entity in entities:
                    if entity.code_system == CodeSystem.NONE:
                        context = text_window(loaded_document.text, entity.span, radius=self.options.context_window)
                        contexts_by_entity[entity.id] = context
                        candidates = self.linker.generate_candidates(entity, context)
                        generated_candidates[entity.id] = candidates
                        generated_total += len(candidates)
                        entities_with_candidates += int(bool(candidates))
                        for candidate in candidates:
                            key = f"source_{candidate.source}"
                            counters[key] = counters.get(key, 0) + 1
                counters["candidate_entities"] = len(generated_candidates)
                counters["entities_with_candidates"] = entities_with_candidates
                counters["generated_candidates"] = generated_total
                counters["candidate_sources"] = len(self.options.candidate_sources)
            else:
                counters["skipped_entities"] = len(entities)

        reranked_candidates: dict[str, list[Candidate]] = {}
        with trace.stage("candidate_reranking") as counters:
            if self.options.enable_linking:
                for entity_id, candidates in generated_candidates.items():
                    if self.options.enable_candidate_reranking:
                        ranked = self.linker.rerank_candidates(candidates, contexts_by_entity.get(entity_id, ""))
                    else:
                        ranked = candidates
                    reranked_candidates[entity_id] = ranked
                counters["reranked_entities"] = len(reranked_candidates)
                counters["reranked_candidates"] = sum(len(candidates) for candidates in reranked_candidates.values())
                counters["skipped_reranking"] = 0 if self.options.enable_candidate_reranking else len(reranked_candidates)
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("normalization_assignment") as counters:
            if self.options.enable_linking:
                assigned_codes = 0
                unlinked_entities = 0
                for entity in entities:
                    assigned_candidates = reranked_candidates.get(entity.id)
                    if assigned_candidates is None:
                        continue
                    self.linker.apply_candidates(entity, assigned_candidates)
                    if entity.code is None:
                        unlinked_entities += 1
                    else:
                        assigned_codes += 1
                counters["assigned_codes"] = assigned_codes
                counters["unlinked_entities"] = unlinked_entities
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
            sentences.extend(split_sentences(section.text, section_title=section.title, base_offset=section.span[0]))
        return sentences or [Sentence(span=(0, len(source_text)), text=source_text)]

    @staticmethod
    def _find_sentence(entity: EntityAnnotation, sentences: list[Sentence]) -> Sentence | None:
        for sentence in sentences:
            if sentence.span[0] <= entity.span[0] and entity.span[1] <= sentence.span[1]:
                return sentence
        return None
