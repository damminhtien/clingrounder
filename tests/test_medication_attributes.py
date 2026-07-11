from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.relations.rule_relations import RuleRelationExtractor
from medical_kg_nlp.schema.types import EntityType, RelationType


def test_medication_attributes_use_dedicated_entity_types_and_relations() -> None:
    text = "Dùng 80mg Lasix IV bid trong 10 ngày."
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entities = RuleBasedNER(store).extract(text)
    by_text = {entity.text: entity for entity in entities}

    assert by_text["80mg"].type == EntityType.STRENGTH
    assert by_text["IV"].type == EntityType.ROUTE
    assert by_text["bid"].type == EntityType.FREQUENCY
    assert by_text["trong 10 ngày"].type == EntityType.DURATION

    relations = RuleRelationExtractor().extract(entities, split_sentences(text))
    relation_types = {(relation.tail, relation.type) for relation in relations}
    assert (by_text["80mg"].id, RelationType.HAS_DOSE) in relation_types
    assert (by_text["IV"].id, RelationType.HAS_ROUTE) in relation_types
    assert (by_text["bid"].id, RelationType.HAS_FREQUENCY) in relation_types
    assert (by_text["trong 10 ngày"].id, RelationType.HAS_DURATION) in relation_types


def test_lab_concentration_near_drug_is_not_misclassified_as_strength() -> None:
    text = "Dùng metformin, glucose 120 mg/dL."
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entities = RuleBasedNER(store).extract(text)

    assert not any(entity.type == EntityType.STRENGTH for entity in entities)
