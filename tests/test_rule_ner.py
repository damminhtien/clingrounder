from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.schema.types import EntityType


def test_rule_ner_applies_data_driven_false_positive_blacklist(tmp_path) -> None:
    blacklist_path = tmp_path / "blacklist.jsonl"
    blacklist_path.write_text(
        '{"alias":"đau bụng","left_regex":"mẫu\\\\s*$","context_radius":20}\n',
        encoding="utf-8",
    )
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entities = RuleBasedNER(store, false_positive_path=blacklist_path).extract("mẫu đau bụng. Đau bụng thật.")

    abdominal_pain = [entity for entity in entities if entity.normalized_text == "đau bụng"]

    assert len(abdominal_pain) == 1
    assert abdominal_pain[0].text == "Đau bụng"


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
    cancer_spans = [
        entity.span for entity in entities if entity.text.lower() == "ung thư đại tràng" and entity.type == EntityType.DISEASE
    ]

    assert cancer_spans == [(50, 67)]
    assert all(entity.text.lower() != "ung thư phôi" for entity in entities)


def test_rule_ner_blocks_spouse_azithromycin_but_keeps_patient_use() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    text = (
        "Vợ có các triệu chứng tương tự, được chẩn đoán là giãn phế quản, "
        "phản ứng tốt với azithromycin. Bệnh nhân được kê azithromycin."
    )
    entities = RuleBasedNER(store).extract(text)
    drug_texts = [entity.text for entity in entities if entity.type == EntityType.DRUG]

    assert drug_texts == ["azithromycin"]


def test_rule_ner_allows_alias_boundary_before_concatenated_uppercase_sentence() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    text = "Chẩn đoán u ác của tuyến tiền liệtAnh ấy đang chờ ghép thận."
    entities = RuleBasedNER(store).extract(text)

    prostate_cancer = [
        entity for entity in entities if entity.type == EntityType.DISEASE and entity.text == "u ác của tuyến tiền liệt"
    ]

    assert len(prostate_cancer) == 1
    assert text[prostate_cancer[0].span[0] : prostate_cancer[0].span[1]] == "u ác của tuyến tiền liệt"


def test_rule_ner_does_not_allow_alias_boundary_inside_lowercase_word() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entities = RuleBasedNER(store).extract("Chẩn đoán u ác của tuyến tiền liệtanh ấy đang chờ ghép thận.")

    assert all(entity.text != "u ác của tuyến tiền liệt" for entity in entities)


def test_rule_ner_keeps_strict_boundary_for_uppercase_abbreviations() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entities = RuleBasedNER(store).extract("Khi đến MICU không có MI. Tiền sử CML.")
    disease_texts = [entity.text for entity in entities if entity.type == EntityType.DISEASE]

    assert "MI" in disease_texts
    assert "CML" in disease_texts
    assert all(entity.span != (8, 10) for entity in entities)
