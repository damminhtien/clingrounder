from medical_kg_nlp.context.assertion import AssertionClassifier
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


def test_family_history_rule() -> None:
    entity, sentence = _entity("Cha bệnh nhân bị ung thư phổi.", "ung thư phổi")
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


def test_possible_clause_does_not_leak_to_confirmed_condition() -> None:
    text = "Nghi viêm phổi, đái tháo đường type 2 đang điều trị metformin."
    sentence = Sentence(span=(0, len(text)), text=text)
    classifier = AssertionClassifier()

    possible = _entity_in_sentence(text, "viêm phổi")
    confirmed = _entity_in_sentence(text, "đái tháo đường type 2")

    assert classifier.classify(possible, sentence) == AssertionStatus.POSSIBLE
    assert classifier.classify(confirmed, sentence) == AssertionStatus.PRESENT
