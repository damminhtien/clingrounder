from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.schema.types import EntityType


def test_rule_ner_does_not_export_medication_dose_as_lab_result() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entities = RuleBasedNER(store).extract("Dùng metoprolol 25mg. HbA1c 7.2%. Creatinine 1.4 mg/dL.")
    by_text = {entity.text: entity for entity in entities}

    assert "25mg" not in by_text
    assert by_text["7.2%"].type == EntityType.LAB_RESULT
    assert by_text["1.4 mg/dL"].type == EntityType.LAB_RESULT
