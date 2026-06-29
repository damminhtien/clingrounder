from medical_kg_nlp.evaluation.context_metrics import context_macro_f1, confusion_matrix
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus, EntityType


def test_context_macro_f1_penalizes_severe_context_error() -> None:
    gold = [
        _entity("E1", AssertionStatus.NEGATED),
        _entity("E2", AssertionStatus.FAMILY),
    ]
    pred = [
        _entity("E1", AssertionStatus.PRESENT),
        _entity("E2", AssertionStatus.FAMILY),
    ]

    assert context_macro_f1(gold, pred) == 0.5
    assert confusion_matrix(gold, pred) == {
        "NEGATED": {"PRESENT": 1},
        "FAMILY": {"FAMILY": 1},
    }


def _entity(entity_id: str, assertion: AssertionStatus) -> EntityAnnotation:
    start = 0 if entity_id == "E1" else 10
    return EntityAnnotation(
        id=entity_id,
        span=(start, start + 5),
        text="dummy",
        normalized_text="dummy",
        type=EntityType.DISEASE,
        assertion=assertion,
    )
