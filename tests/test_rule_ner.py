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


def test_rule_ner_blocks_phase1_history_of_ho_artifact_before_disease_name() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    text = "Các bệnh lý mạn tính - ho Rung nhĩ. Triệu chứng hiện tại - ho khan, ho đánh thức và ho mạn tính."
    entities = RuleBasedNER(store).extract(text)
    cough_spans = [entity.span for entity in entities if entity.text.lower() == "ho" and entity.type == EntityType.SYMPTOM]

    assert cough_spans == [(59, 61), (68, 70), (84, 86)]
    assert any(entity.text == "Rung nhĩ" and entity.type == EntityType.DISEASE for entity in entities)


def test_rule_ner_blocks_cancer_inside_cea_lab_name_but_keeps_real_cancer_mentions() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    text = "CEA (kháng nguyên ung thư phôi) tăng nhẹ. Tiền sử ung thư đại tràng."
    entities = RuleBasedNER(store).extract(text)
    cancer_spans = [entity.span for entity in entities if entity.text.lower() == "ung thư" and entity.type == EntityType.DISEASE]

    assert cancer_spans == [(50, 57)]
