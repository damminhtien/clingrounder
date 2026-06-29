from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medical_kg_nlp.context.assertion import AssertionClassifier
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.kg.validator import KGValidator
from medical_kg_nlp.linking.linker import EntityLinker
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.pipeline.tracing import PipelineTrace
from medical_kg_nlp.preprocessing.section_splitter import split_sections
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.relations.rule_relations import RuleRelationExtractor
from medical_kg_nlp.retrieval.candidate_generator import CandidateGenerator
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument, Sentence
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

    def process_document(self, document: ClinicalDocument) -> ClinicalPrediction:
        return self.process_document_with_trace(document).prediction

    def process_document_with_trace(self, document: ClinicalDocument) -> PipelineRunResult:
        trace = PipelineTrace(document_id=document.document_id)
        with trace.stage("sentence_split") as counters:
            sentences = self._sentences(document.text)
            counters["sentences"] = len(sentences)

        with trace.stage("rule_ner") as counters:
            entities = self.ner.extract(document.text)
            counters["entities"] = len(entities)

        with trace.stage("context_assertion") as counters:
            if self.options.enable_context:
                for entity in entities:
                    sentence = self._find_sentence(entity, sentences)
                    entity.assertion = self.assertion.classify(entity, sentence)
                counters["classified_entities"] = len(entities)
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("entity_linking") as counters:
            if self.options.enable_linking:
                linked_entities = 0
                for entity in entities:
                    if entity.code_system == CodeSystem.NONE:
                        context = text_window(document.text, entity.span, radius=self.options.context_window)
                        self.linker.link_entity(entity, context)
                        linked_entities += 1
                counters["linked_entities"] = linked_entities
                counters["candidate_sources"] = len(self.options.candidate_sources)
            else:
                counters["skipped_entities"] = len(entities)

        with trace.stage("entity_kg_validation") as counters:
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

        with trace.stage("relation_kg_validation") as counters:
            if self.options.enable_relation_kg_validation:
                relations, relation_issues = self.validator.validate_relations(entities, relations)
                counters["issues"] = len(relation_issues)
                counters["relations"] = len(relations)
            else:
                counters["relations"] = len(relations)

        with trace.stage("prediction_validation") as counters:
            prediction = ClinicalPrediction.from_text(
                document_id=document.document_id,
                text=document.text,
                entities=entities,
                relations=relations,
                pipeline_version=self.pipeline_version,
            )
            prediction.validate(document.text)
            counters["entities"] = len(prediction.entities)
            counters["relations"] = len(prediction.relations)
        return PipelineRunResult(prediction=prediction, trace=trace)

    def _sentences(self, text: str) -> list[Sentence]:
        sentences: list[Sentence] = []
        for section in split_sections(text):
            sentences.extend(split_sentences(section.text, section_title=section.title, base_offset=section.span[0]))
        return sentences or [Sentence(span=(0, len(text)), text=text)]

    @staticmethod
    def _find_sentence(entity: EntityAnnotation, sentences: list[Sentence]) -> Sentence | None:
        for sentence in sentences:
            if sentence.span[0] <= entity.span[0] and entity.span[1] <= sentence.span[1]:
                return sentence
        return None
