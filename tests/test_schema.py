import math

import pytest

from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.annotation import RelationAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType, RelationType


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


@pytest.mark.parametrize("span", [(0, 0), (-1, 1), (3, 2)])
def test_entity_rejects_invalid_spans(span: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="span"):
        EntityAnnotation("E1", span, "x", "x", EntityType.SYMPTOM)


def test_entity_rejects_nonfinite_confidence_and_code_mismatch() -> None:
    with pytest.raises(ValueError, match="confidence"):
        EntityAnnotation("E1", (0, 1), "x", "x", EntityType.SYMPTOM, confidence=math.nan)
    with pytest.raises(ValueError, match="requires a null"):
        EntityAnnotation(
            "E1", (0, 1), "x", "x", EntityType.SYMPTOM, code_system=CodeSystem.NONE, code="x"
        )


def test_candidate_rejects_inconsistent_code_state() -> None:
    with pytest.raises(ValueError, match="requires a null"):
        Candidate(
            concept_id="C1",
            code="I10",
            code_system=CodeSystem.NONE,
            canonical_name="condition",
            semantic_type=EntityType.DISEASE,
            score=0.5,
            source="test",
        )


def test_relation_rejects_self_loop_and_nonfinite_confidence() -> None:
    with pytest.raises(ValueError, match="different"):
        RelationAnnotation("R1", "E1", "E1", RelationType.ASSOCIATED_WITH, 0.5)
    with pytest.raises(ValueError, match="confidence"):
        RelationAnnotation("R1", "E1", "E2", RelationType.ASSOCIATED_WITH, math.inf)


def test_document_requires_nonempty_identity_and_text() -> None:
    with pytest.raises(ValueError, match="document_id"):
        ClinicalDocument("", "text")
    with pytest.raises(ValueError, match="text"):
        ClinicalDocument("doc", "")
