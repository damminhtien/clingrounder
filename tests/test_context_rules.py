from medical_kg_nlp.context.assertion import AssertionClassifier
from medical_kg_nlp.context.rules import PLANNED_LEFT_CUES, POSSIBLE_LEFT_CUES, POSSIBLE_RIGHT_CUES
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType


def _entity(text: str, mention: str) -> tuple[EntityAnnotation, Sentence]:
    start = text.index(mention)
    return (
        EntityAnnotation(
            id="E1",
            span=(start, start + len(mention)),
            text=mention,
            normalized_text=mention.lower(),
            type=EntityType.DISEASE,
            code_system=CodeSystem.NONE,
        ),
        Sentence(span=(0, len(text)), text=text),
    )


def _entity_in_sentence(text: str, mention: str) -> EntityAnnotation:
    start = text.index(mention)
    return EntityAnnotation(
        id=mention,
        span=(start, start + len(mention)),
        text=mention,
        normalized_text=mention.lower(),
        type=EntityType.DISEASE,
        code_system=CodeSystem.NONE,
    )


def test_negation_rule() -> None:
    entity, sentence = _entity("Không ghi nhận viêm phổi.", "viêm phổi")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.NEGATED


def test_source_backed_negation_cue() -> None:
    entity, sentence = _entity("Patient is free of chest pain.", "chest pain")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.NEGATED


def test_possible_rule_overrides_negation_phrase() -> None:
    entity, sentence = _entity("Không loại trừ viêm phổi.", "viêm phổi")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.POSSIBLE


def test_bidirectional_possible_cue_is_applied_on_the_right() -> None:
    entity, sentence = _entity("Viêm phổi không loại trừ", "Viêm phổi")

    assert "không loại trừ" in POSSIBLE_LEFT_CUES
    assert "không loại trừ" in POSSIBLE_RIGHT_CUES
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.POSSIBLE


def test_section_prior_cue_is_not_loaded_as_left_context_cue() -> None:
    assert "plan" not in PLANNED_LEFT_CUES


def test_vietnamese_ruled_out_cue_is_negated_but_not_khong_loai_tru() -> None:
    ruled_out, ruled_out_sentence = _entity("Đã loại trừ nhồi máu cơ tim.", "nhồi máu cơ tim")
    possible, possible_sentence = _entity("Không loại trừ nhồi máu cơ tim.", "nhồi máu cơ tim")

    classifier = AssertionClassifier()

    assert classifier.classify(ruled_out, ruled_out_sentence) == AssertionStatus.NEGATED
    assert classifier.classify(possible, possible_sentence) == AssertionStatus.POSSIBLE


def test_non_specific_phrase_does_not_negate_condition() -> None:
    entity, sentence = _entity("Hình ảnh không đặc hiệu cho viêm phổi.", "viêm phổi")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.PRESENT


def test_non_specific_phrase_does_not_mask_real_negation_cue() -> None:
    entity, sentence = _entity("Không ghi nhận viêm phổi, hình ảnh không đặc hiệu.", "viêm phổi")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.NEGATED


def test_family_history_rule() -> None:
    entity, sentence = _entity("Cha bệnh nhân bị ung thư phổi.", "ung thư phổi")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.FAMILY


def test_family_member_observation_does_not_mark_patient_condition_as_family() -> None:
    entity, sentence = _entity("Con trai phát hiện bệnh nhân có viêm phổi tại nhà.", "viêm phổi")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.PRESENT


def test_family_member_with_disease_predicate_still_marks_family_history() -> None:
    entity, sentence = _entity("Anh bệnh nhân bị ung thư phổi.", "ung thư phổi")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.FAMILY


def test_source_backed_family_cue() -> None:
    entity, sentence = _entity("Maternal history of asthma.", "asthma")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.FAMILY


def test_family_history_clause_does_not_leak_to_patient_condition() -> None:
    text = "Cha bệnh nhân bị ung thư phổi, bệnh nhân có đái tháo đường type 2."
    sentence = Sentence(span=(0, len(text)), text=text)
    classifier = AssertionClassifier()

    family = _entity_in_sentence(text, "ung thư phổi")
    patient_condition = _entity_in_sentence(text, "đái tháo đường type 2")

    assert classifier.classify(family, sentence) == AssertionStatus.FAMILY
    assert classifier.classify(patient_condition, sentence) == AssertionStatus.PRESENT


def test_negation_clause_does_not_leak_to_present_condition() -> None:
    text = "Không ghi nhận viêm phổi, bệnh nhân có đái tháo đường type 2."
    sentence = Sentence(span=(0, len(text)), text=text)
    classifier = AssertionClassifier()

    negated = _entity_in_sentence(text, "viêm phổi")
    present = _entity_in_sentence(text, "đái tháo đường type 2")

    assert classifier.classify(negated, sentence) == AssertionStatus.NEGATED
    assert classifier.classify(present, sentence) == AssertionStatus.PRESENT


def test_negation_scope_covers_same_sentence_vietnamese_symptom_list() -> None:
    text = "Bệnh nhân không choáng váng, chóng mặt, buồn nôn, đánh trống ngực, hoặc đau ngực."
    sentence = Sentence(span=(0, len(text)), text=text)
    classifier = AssertionClassifier()

    for mention in ("choáng váng", "chóng mặt", "buồn nôn", "đánh trống ngực", "đau ngực"):
        entity = _entity_in_sentence(text, mention)
        entity.type = EntityType.SYMPTOM
        assert classifier.classify(entity, sentence) == AssertionStatus.NEGATED


def test_negation_coordination_stops_at_present_patient_clause() -> None:
    text = "Không ghi nhận viêm phổi, bệnh nhân có đái tháo đường type 2."
    sentence = Sentence(span=(0, len(text)), text=text)
    classifier = AssertionClassifier()

    present = _entity_in_sentence(text, "đái tháo đường type 2")
    assert classifier.classify(present, sentence) == AssertionStatus.PRESENT


def test_negation_from_intolerance_does_not_leak_to_switched_medication() -> None:
    text = "Bệnh nhân không dung nạp amoxicillin do tiêu chảy nên được chuyển sang sử dụng azithromycin."
    sentence = Sentence(span=(0, len(text)), text=text)
    classifier = AssertionClassifier()

    stopped = _entity_in_sentence(text, "amoxicillin")
    stopped.type = EntityType.DRUG
    switched = _entity_in_sentence(text, "azithromycin")
    switched.type = EntityType.DRUG

    assert classifier.classify(stopped, sentence) == AssertionStatus.NEGATED
    assert classifier.classify(switched, sentence) == AssertionStatus.PRESENT


def test_non_disease_negative_phrases_do_not_negate_later_diagnosis() -> None:
    text = "Không thể giữ được bất cứ thứ gì, chẩn đoán viêm dạ dày ruột do virus."
    sentence = Sentence(span=(0, len(text)), text=text)
    entity = _entity_in_sentence(text, "viêm dạ dày ruột do virus")

    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.PRESENT


def test_without_contrast_imaging_phrase_does_not_negate_suggested_diagnosis() -> None:
    text = (
        "Chụp CT bụng, chậu, không thuốc cản quang cho thấy dịch quanh đại tràng sigma, "
        "gợi ý viêm túi mật cấp tính không biến chứng."
    )
    sentence = Sentence(span=(0, len(text)), text=text)
    entity = _entity_in_sentence(text, "viêm túi mật cấp tính không biến chứng")

    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.POSSIBLE


def test_possible_reset_stops_negation_from_leaking_to_likely_diagnosis() -> None:
    text = "Không nghĩ đây là biến đổi cấp tính và có khả năng đại diện cho CML trong bối cảnh ngừng thuốc."
    sentence = Sentence(span=(0, len(text)), text=text)
    entity = _entity_in_sentence(text, "CML")

    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.POSSIBLE


def test_postoperative_patient_condition_breaks_earlier_negated_procedure_clause() -> None:
    text = (
        "ERCP không thể qua được chỗ gián đoạn ống dẫn, không làm giảm đáng kể lượng dịch rò, "
        "hậu phẫu bệnh nhân bị nhiễm Clostridioides difficile."
    )
    sentence = Sentence(span=(0, len(text)), text=text)
    entity = _entity_in_sentence(text, "nhiễm Clostridioides difficile")

    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.PRESENT


def test_possible_clause_does_not_leak_to_confirmed_condition() -> None:
    text = "Nghi viêm phổi, đái tháo đường type 2 đang điều trị metformin."
    sentence = Sentence(span=(0, len(text)), text=text)
    classifier = AssertionClassifier()

    possible = _entity_in_sentence(text, "viêm phổi")
    confirmed = _entity_in_sentence(text, "đái tháo đường type 2")

    assert classifier.classify(possible, sentence) == AssertionStatus.POSSIBLE
    assert classifier.classify(confirmed, sentence) == AssertionStatus.PRESENT


def test_historical_section_prior_overrides_possible_cue_for_phase1_pmh() -> None:
    text = "nghi ngờ xơ gan do rượu"
    sentence = Sentence(span=(0, len(text)), text=text, section_title="Các bệnh lý mạn tính")
    entity = _entity_in_sentence(text, "xơ gan do rượu")

    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.HISTORICAL


def test_negation_still_overrides_historical_section_prior() -> None:
    text = "không có viêm phổi"
    sentence = Sentence(span=(0, len(text)), text=text, section_title="Các bệnh lý mạn tính")
    entity = _entity_in_sentence(text, "viêm phổi")

    classifier = AssertionClassifier()
    features = classifier.classify_features(entity, sentence)

    assert classifier.classify(entity, sentence) == AssertionStatus.NEGATED
    assert features.negated is True
    assert features.historical is True


def test_lab_test_can_be_negated_or_planned() -> None:
    negated, negated_sentence = _typed_entity(
        "Chưa thực hiện xét nghiệm CRP.",
        "CRP",
        EntityType.LAB_TEST,
    )
    planned, planned_sentence = _typed_entity(
        "Dự kiến xét nghiệm CRP ngày mai.",
        "CRP",
        EntityType.LAB_TEST,
    )

    classifier = AssertionClassifier()
    assert classifier.classify(negated, negated_sentence) == AssertionStatus.NEGATED
    assert classifier.classify(planned, planned_sentence) == AssertionStatus.PLANNED


def test_lab_result_can_be_negated_or_historical() -> None:
    negated, negated_sentence = _typed_entity(
        "Không ghi nhận glucose tăng.",
        "tăng",
        EntityType.LAB_RESULT,
    )
    historical, historical_sentence = _typed_entity(
        "Tiền sử HbA1c tăng.",
        "tăng",
        EntityType.LAB_RESULT,
    )

    classifier = AssertionClassifier()
    assert classifier.classify(negated, negated_sentence) == AssertionStatus.NEGATED
    assert classifier.classify(historical, historical_sentence) == AssertionStatus.HISTORICAL


def _typed_entity(
    text: str,
    mention: str,
    entity_type: EntityType,
) -> tuple[EntityAnnotation, Sentence]:
    entity, sentence = _entity(text, mention)
    entity.type = entity_type
    return entity, sentence
