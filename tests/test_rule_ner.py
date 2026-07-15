import time
from pathlib import Path

import pytest

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_rule_ner_pins_unique_exact_dictionary_code() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entity = next(
        item
        for item in RuleBasedNER(
            store,
            emit_probabilities_by_source={"RxNorm:dictionary_exact": 0.97},
        ).extract("Dùng metformin.")
        if item.text == "metformin"
    )

    assert entity.code_system == CodeSystem.RXNORM
    assert entity.code == "6809"
    assert entity.candidates[0].source == "dictionary_exact"
    assert entity.candidates[0].emit_probability == 0.97


def test_rule_ner_abstains_from_uncalibrated_dictionary_candidate() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entity = next(
        item for item in RuleBasedNER(store).extract("Dùng metformin.") if item.text == "metformin"
    )

    assert entity.candidates[0].emit_probability == 0.0


def test_rule_ner_abstains_when_exact_alias_maps_to_multiple_codes() -> None:
    entries = [
        ConceptEntry(
            concept_id=f"D:{code}",
            code=code,
            code_system=CodeSystem.ICD10,
            canonical_name=name,
            semantic_type=EntityType.DISEASE,
            aliases=("bệnh mơ hồ",),
        )
        for code, name in (("A00", "Disease A"), ("B00", "Disease B"))
    ]
    entity = RuleBasedNER(DictionaryStore(entries)).extract("Có bệnh mơ hồ.")[0]

    assert entity.code_system == CodeSystem.NONE
    assert entity.code is None
    assert entity.candidates == []


def test_rule_ner_skips_exact_alias_with_cross_type_ambiguity() -> None:
    entries = [
        ConceptEntry(
            concept_id="D:A00",
            code="A00",
            code_system=CodeSystem.ICD10,
            canonical_name="Disease ambiguous",
            semantic_type=EntityType.DISEASE,
            aliases=("khái niệm mơ hồ",),
        ),
        ConceptEntry(
            concept_id="RX:1",
            code="1",
            code_system=CodeSystem.RXNORM,
            canonical_name="Drug ambiguous",
            semantic_type=EntityType.DRUG,
            aliases=("khái niệm mơ hồ",),
        ),
    ]

    entities = RuleBasedNER(DictionaryStore(entries)).extract("Có khái niệm mơ hồ.")

    assert entities == []


def test_rule_ner_applies_data_driven_false_positive_blacklist(tmp_path) -> None:
    blacklist_path = tmp_path / "blacklist.jsonl"
    blacklist_path.write_text(
        '{"alias":"đau bụng","left_regex":"mẫu\\\\s*$","context_radius":20}\n',
        encoding="utf-8",
    )
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entities = RuleBasedNER(store, false_positive_path=blacklist_path).extract(
        "mẫu đau bụng. Đau bụng thật."
    )

    abdominal_pain = [entity for entity in entities if entity.normalized_text == "đau bụng"]

    assert len(abdominal_pain) == 1
    assert abdominal_pain[0].text == "Đau bụng"


def test_rule_ner_classifies_medication_strength_separately_from_lab_result() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    text = "Dùng metoprolol 25mg. HbA1c 7.2%. Creatinine 1.4 mg/dL."
    entities = RuleBasedNER(store).extract(text)
    by_text = {entity.text: entity for entity in entities}

    assert by_text["25mg"].type == EntityType.STRENGTH
    medication = by_text["metoprolol"].medication_mention
    assert medication is not None
    assert text[medication.full_span[0] : medication.full_span[1]] == "metoprolol 25mg"
    assert by_text["7.2%"].type == EntityType.LAB_RESULT
    assert by_text["1.4 mg/dL"].type == EntityType.LAB_RESULT
def test_rule_ner_prefers_combination_drug_over_overlapping_ingredients() -> None:
    entries = [
        ConceptEntry(
            concept_id="RX:AMOX",
            code="1",
            code_system=CodeSystem.RXNORM,
            canonical_name="amoxicillin",
            semantic_type=EntityType.DRUG,
        ),
        ConceptEntry(
            concept_id="RX:CLAV",
            code="2",
            code_system=CodeSystem.RXNORM,
            canonical_name="clavulanate",
            semantic_type=EntityType.DRUG,
        ),
        ConceptEntry(
            concept_id="RX:COMBO",
            code="3",
            code_system=CodeSystem.RXNORM,
            canonical_name="amoxicillin / clavulanate",
            semantic_type=EntityType.DRUG,
            aliases=("amoxicillin/clavulanate",),
        ),
    ]
    text = "Dùng amoxicillin/clavulanate 875 mg tablet po bid."

    entities = RuleBasedNER(DictionaryStore(entries)).extract(text)
    drugs = [entity for entity in entities if entity.type == EntityType.DRUG]

    assert [(entity.text, entity.code) for entity in drugs] == [
        ("amoxicillin/clavulanate", "3")
    ]
    medication = drugs[0].medication_mention
    assert medication is not None
    assert text[medication.full_span[0] : medication.full_span[1]] == (
        "amoxicillin/clavulanate 875 mg tablet po bid"
    )


def test_rule_ner_extracts_vietnamese_vital_sign_names_and_value_spans() -> None:
    store = DictionaryStore.from_jsonl(
        "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl"
    )
    text = (
        "Dấu hiệu sinh tồn: huyết áp 159/72, nhịp thở 20, nhịp tim 84, "
        "SpO2 96%, nhiệt độ 36.7°c. Tăng huyết áp đang điều trị."
    )
    entities = RuleBasedNER(store).extract(text)
    by_text = {entity.text: entity for entity in entities}

    assert by_text["huyết áp"].type == EntityType.LAB_TEST
    assert by_text["159/72"].type == EntityType.LAB_RESULT
    assert by_text["nhịp thở"].type == EntityType.LAB_TEST
    assert by_text["20"].type == EntityType.LAB_RESULT
    assert by_text["nhịp tim"].type == EntityType.LAB_TEST
    assert by_text["84"].type == EntityType.LAB_RESULT
    assert by_text["SpO2"].type == EntityType.LAB_TEST
    assert by_text["96%"].type == EntityType.LAB_RESULT
    assert by_text["nhiệt độ"].type == EntityType.LAB_TEST
    assert by_text["36.7°c"].type == EntityType.LAB_RESULT
    assert by_text["Tăng huyết áp"].type == EntityType.DISEASE
    for entity in entities:
        assert text[entity.span[0] : entity.span[1]] == entity.text


def test_rule_ner_extracts_bare_and_qualitative_results_only_after_lab_anchors() -> None:
    store = DictionaryStore.from_jsonl(
        "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl"
    )
    text = "Kali: 2.4; creatinine (serum) = 1.3. Chụp CT không ghi nhận gì bất thường. Số trần 42."

    entities = RuleBasedNER(store).extract(text)
    lab_results = [entity for entity in entities if entity.type == EntityType.LAB_RESULT]
    result_texts = [entity.text for entity in lab_results]

    assert "2.4" in result_texts
    assert "1.3" in result_texts
    assert "không ghi nhận gì bất thường" in result_texts
    assert "42" not in result_texts
    for entity in lab_results:
        assert text[entity.span[0] : entity.span[1]] == entity.text


def test_rule_ner_does_not_treat_dates_or_drug_doses_as_anchored_lab_results() -> None:
    store = DictionaryStore.from_jsonl(
        "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl"
    )
    text = "Creatinine ngày 12/07/2026. Dùng metoprolol 25 mg. Kali được bổ sung 40 mg."

    entities = RuleBasedNER(store).extract(text)
    lab_result_texts = [entity.text for entity in entities if entity.type == EntityType.LAB_RESULT]

    assert "12/07/2026" not in lab_result_texts
    assert "25" not in lab_result_texts
    assert "40" not in lab_result_texts


def test_rule_ner_extracts_qualitative_results_before_or_after_lab_anchor() -> None:
    store = DictionaryStore.from_jsonl(
        "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl"
    )
    text = "Glucose là thấp. tăng Cr theo phòng cấp cứu. Kali tăng cao."

    entities = RuleBasedNER(store).extract(text)
    lab_result_texts = [entity.text for entity in entities if entity.type == EntityType.LAB_RESULT]

    assert "thấp" in lab_result_texts
    assert "tăng" in lab_result_texts
    assert "tăng cao" in lab_result_texts


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
    cough_spans = [
        entity.span
        for entity in entities
        if entity.text.lower() == "ho" and entity.type == EntityType.SYMPTOM
    ]

    assert cough_spans == [(59, 61), (68, 70), (84, 86)]
    assert any(
        entity.text == "Rung nhĩ" and entity.type == EntityType.DISEASE for entity in entities
    )


def test_rule_ner_blocks_cancer_inside_cea_lab_name_but_keeps_real_cancer_mentions() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    text = "CEA (kháng nguyên ung thư phôi) tăng nhẹ. Tiền sử ung thư đại tràng."
    entities = RuleBasedNER(store).extract(text)
    cancer_spans = [
        entity.span
        for entity in entities
        if entity.text.lower() == "ung thư đại tràng" and entity.type == EntityType.DISEASE
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
        entity
        for entity in entities
        if entity.type == EntityType.DISEASE and entity.text == "u ác của tuyến tiền liệt"
    ]

    assert len(prostate_cancer) == 1
    assert (
        text[prostate_cancer[0].span[0] : prostate_cancer[0].span[1]] == "u ác của tuyến tiền liệt"
    )


def test_rule_ner_does_not_allow_alias_boundary_inside_lowercase_word() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entities = RuleBasedNER(store).extract(
        "Chẩn đoán u ác của tuyến tiền liệtanh ấy đang chờ ghép thận."
    )

    assert all(entity.text != "u ác của tuyến tiền liệt" for entity in entities)


def test_rule_ner_keeps_strict_boundary_for_uppercase_abbreviations() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entities = RuleBasedNER(store).extract("Khi đến MICU không có MI. Tiền sử CML.")
    disease_texts = [entity.text for entity in entities if entity.type == EntityType.DISEASE]

    assert "MI" in disease_texts
    assert "CML" in disease_texts
    assert all(entity.span != (8, 10) for entity in entities)


def test_rule_ner_recovers_concatenated_drug_aliases_with_source_offsets() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    text = (
        "Dùngmethadonekéo dài. "
        "Tiếp tục doxycyclinebactrim. "
        "lasixđã dừng. "
        "morphineiv morphineoral. "
        "klonopinclonidine và ciproflagyl. "
        "atenololtrong ngày."
    )
    entities = RuleBasedNER(store).extract(text)
    drug_texts = [entity.text for entity in entities if entity.type == EntityType.DRUG]

    for expected in (
        "methadone",
        "doxycycline",
        "bactrim",
        "lasix",
        "morphine",
        "klonopin",
        "clonidine",
        "cipro",
        "flagyl",
        "atenolol",
    ):
        assert expected in [item.lower() for item in drug_texts]
    for entity in entities:
        assert text[entity.span[0] : entity.span[1]] == entity.text


def test_rule_ner_uses_medication_indication_context_for_dual_typed_concept() -> None:
    store = DictionaryStore(
        [
            ConceptEntry(
                concept_id="D:1",
                code="R42",
                code_system=CodeSystem.ICD10,
                canonical_name="chóng mặt",
                semantic_type=EntityType.DISEASE,
            ),
            ConceptEntry(
                concept_id="S:1",
                code=None,
                code_system=CodeSystem.NONE,
                canonical_name="chóng mặt",
                semantic_type=EntityType.SYMPTOM,
            ),
            ConceptEntry(
                concept_id="RX:1",
                code="1",
                code_system=CodeSystem.RXNORM,
                canonical_name="thuốc thử",
                semantic_type=EntityType.DRUG,
            ),
        ]
    )
    ner = RuleBasedNER(store)

    indication = ner.extract("1. thuốc thử 5 mg po daily điều trị chóng mặt")
    outside_list = ner.extract("Bệnh nhân có chóng mặt.")

    assert next(entity for entity in indication if entity.text == "chóng mặt").type == (
        EntityType.SYMPTOM
    )
    assert next(entity for entity in outside_list if entity.text == "chóng mặt").type == (
        EntityType.DISEASE
    )


@pytest.mark.benchmark
@pytest.mark.private
def test_rule_ner_phase1_latency_under_100ms_per_note() -> None:
    store = DictionaryStore.from_jsonl(
        "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl"
    )
    ner = RuleBasedNER(store)
    texts = [
        path.read_text(encoding="utf-8")
        for path in sorted(Path("data/raw/input").glob("*.txt"), key=lambda item: int(item.stem))
    ]

    started = time.perf_counter()
    entity_count = sum(len(ner.extract(text)) for text in texts)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    average_ms = elapsed_ms / len(texts)

    assert entity_count > 0
    assert average_ms < 100.0
