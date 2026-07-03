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


def test_rule_ner_blocks_ambiguous_yeu_to_and_chu_yeu_contexts() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entities = RuleBasedNER(store).extract(
        "Các yếu tố làm nặng thêm: gắng sức. Đau chủ yếu sau ăn. Bệnh nhân mệt mỏi, yếu cơ và yếu sức.",
    )
    by_text = {entity.text: entity for entity in entities}

    assert "yếu" not in by_text
    assert by_text["mệt mỏi"].type == EntityType.SYMPTOM
    assert by_text["yếu cơ"].type == EntityType.SYMPTOM
    assert by_text["yếu sức"].type == EntityType.SYMPTOM


def test_rule_ner_prefers_long_phase1_symptom_spans() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entities = RuleBasedNER(store).extract("Có khó thở khi gắng sức và phù mắt cá chân.")
    by_text = {entity.text: entity for entity in entities}

    assert by_text["khó thở khi gắng sức"].type == EntityType.SYMPTOM
    assert by_text["phù mắt cá chân"].type == EntityType.SYMPTOM
    assert "khó thở" not in by_text
    assert "phù" not in by_text
