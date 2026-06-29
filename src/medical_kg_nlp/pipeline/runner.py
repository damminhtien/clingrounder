from __future__ import annotations
from pathlib import Path

from medical_kg_nlp.context.assertion import AssertionClassifier
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.kg.validator import KGValidator
from medical_kg_nlp.linking.linker import EntityLinker
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.preprocessing.section_splitter import split_sections
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.relations.rule_relations import RuleRelationExtractor
from medical_kg_nlp.retrieval.candidate_generator import CandidateGenerator
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument, Sentence
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import CodeSystem
from medical_kg_nlp.utils.text import text_window


class PipelineRunner:
    def __init__(
        self,
        dictionary_path: str | Path = "data/dictionaries/seed_concepts.jsonl",
        abbreviation_path: str | Path = "data/dictionaries/abbreviations.jsonl",
        pipeline_version: str = "0.1.0",
    ) -> None:
        self.store = DictionaryStore.from_jsonl(dictionary_path)
        self.ner = RuleBasedNER(self.store)
        self.linker = EntityLinker(CandidateGenerator(self.store, abbreviation_path=abbreviation_path))
        self.assertion = AssertionClassifier()
        self.relations = RuleRelationExtractor()
        self.validator = KGValidator()
        self.pipeline_version = pipeline_version

    def process_document(self, document: ClinicalDocument) -> ClinicalPrediction:
        sentences = self._sentences(document.text)
        sentence_for_entity: dict[str, Sentence] = {}
        entities = self.ner.extract(document.text)
        for entity in entities:
            sentence = self._find_sentence(entity, sentences)
            if sentence is not None:
                sentence_for_entity[entity.id] = sentence
            entity.assertion = self.assertion.classify(entity, sentence)
            if entity.code_system == CodeSystem.NONE:
                self.linker.link_entity(entity, text_window(document.text, entity.span))
        entities, _entity_issues = self.validator.validate_entities(entities)
        relations = self.relations.extract(entities, sentences)
        relations, _relation_issues = self.validator.validate_relations(entities, relations)
        prediction = ClinicalPrediction.from_text(
            document_id=document.document_id,
            text=document.text,
            entities=entities,
            relations=relations,
            pipeline_version=self.pipeline_version,
        )
        prediction.validate(document.text)
        return prediction

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

