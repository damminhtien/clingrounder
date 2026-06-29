from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType


def test_prediction_validates_exact_offsets() -> None:
    text = "metformin"
    entity = EntityAnnotation(
        id="E1",
        span=(0, 9),
        text="metformin",
        normalized_text="metformin",
        type=EntityType.DRUG,
        assertion=AssertionStatus.PRESENT,
        code_system=CodeSystem.RXNORM,
        code="6809",
        confidence=1.0,
    )
    prediction = ClinicalPrediction.from_text("doc", text, [entity], [], "test")
    prediction.validate(text)
    assert prediction.to_json()["entities"][0]["code"] == "6809"

