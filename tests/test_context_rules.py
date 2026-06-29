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


def test_negation_rule() -> None:
    entity, sentence = _entity("Không ghi nhận viêm phổi.", "viêm phổi")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.NEGATED


def test_possible_rule_overrides_negation_phrase() -> None:
    entity, sentence = _entity("Không loại trừ viêm phổi.", "viêm phổi")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.POSSIBLE


def test_family_history_rule() -> None:
    entity, sentence = _entity("Cha bệnh nhân bị ung thư phổi.", "ung thư phổi")
    assert AssertionClassifier().classify(entity, sentence) == AssertionStatus.FAMILY

