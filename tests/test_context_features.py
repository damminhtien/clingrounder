"""Assertion model features remain independent from assertion decoding."""

from medical_kg_nlp.context import AssertionClassifier, AssertionModelFeatureExtractor
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_assertion_features_include_section_rule_and_independent_attribute() -> None:
    text = "Không ghi nhận viêm phổi."
    sentence = Sentence(
        span=(0, len(text)),
        text=text,
        section_title="Tiền sử bệnh",
    )
    start = text.index("viêm phổi")
    entity = EntityAnnotation(
        id="E1",
        span=(start, start + len("viêm phổi")),
        text="viêm phổi",
        normalized_text="viêm phổi",
        type=EntityType.DISEASE,
        code_system=CodeSystem.NONE,
    )
    decisions, graph = AssertionClassifier().classify_batch_with_graph(
        [entity],
        sentence,
    )

    features = AssertionModelFeatureExtractor().extract(entity, sentence, graph)

    assert decisions["E1"][0].negated is True
    assert decisions["E1"][0].historical is True
    assert features["assertion:NEGATED"] == 1.0
    assert features["assertion:HISTORICAL"] == 1.0
    assert features["section:tiền sử bệnh"] == 1.0
    assert features["modifier_count"] == 2.0
    assert sum(name.startswith("rule:") for name in features) == 2


def test_assertion_features_mark_absent_modifier_without_inventing_status() -> None:
    text = "Bệnh nhân có đau ngực."
    sentence = Sentence(span=(0, len(text)), text=text)
    start = text.index("đau ngực")
    entity = EntityAnnotation(
        id="E1",
        span=(start, start + len("đau ngực")),
        text="đau ngực",
        normalized_text="đau ngực",
        type=EntityType.SYMPTOM,
        code_system=CodeSystem.NONE,
    )
    _, graph = AssertionClassifier().classify_batch_with_graph([entity], sentence)

    features = AssertionModelFeatureExtractor().extract(entity, sentence, graph)

    assert features["no_modifier"] == 1.0
    assert features["modifier_count"] == 0.0
    assert not any(name.startswith("assertion:") for name in features)
