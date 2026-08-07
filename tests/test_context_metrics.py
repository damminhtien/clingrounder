import pytest

from clingrounder.evaluation.context_metrics import (
    assertion_attribute_metrics,
    context_macro_f1,
    confusion_matrix,
)
from clingrounder.schema.annotation import AssertionFeatures, EntityAnnotation
from clingrounder.schema.types import AssertionStatus, EntityType


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


def test_assertion_attribute_metrics_score_dimensions_independently() -> None:
    gold = [
        _entity(
            "E1",
            AssertionStatus.NEGATED,
            assertion_features=AssertionFeatures(negated=True, historical=True),
        ),
        _entity("E2", AssertionStatus.FAMILY),
    ]
    pred = [
        _entity(
            "E1",
            AssertionStatus.NEGATED,
            assertion_features=AssertionFeatures(negated=True),
        ),
        _entity("E2", AssertionStatus.PRESENT),
    ]

    report = assertion_attribute_metrics(gold, pred)

    assert report["matched_entity_count"] == 2
    assert report["missing_gold_count"] == 0
    assert report["spurious_prediction_count"] == 0
    assert report["active_attribute_macro_f1"] == pytest.approx(1 / 3)
    assert report["attributes"]["negated"] == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "tn": 1,
        "support": 1,
        "predicted_positive": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert report["attributes"]["historical"]["fn"] == 1
    assert report["attributes"]["family"]["fn"] == 1


def test_assertion_attribute_metrics_report_projection_changes_separately() -> None:
    gold = [_entity("E1", AssertionStatus.NEGATED)]
    pred = [_entity("E2", AssertionStatus.NEGATED)]

    report = assertion_attribute_metrics(gold, pred)

    assert report["matched_entity_count"] == 0
    assert report["missing_gold_count"] == 1
    assert report["spurious_prediction_count"] == 1
    assert report["attributes"]["negated"]["support"] == 0


def _entity(
    entity_id: str,
    assertion: AssertionStatus,
    *,
    assertion_features: AssertionFeatures | None = None,
) -> EntityAnnotation:
    start = 0 if entity_id == "E1" else 10
    return EntityAnnotation(
        id=entity_id,
        span=(start, start + 5),
        text="dummy",
        normalized_text="dummy",
        type=EntityType.DISEASE,
        assertion=assertion,
        assertion_features=assertion_features or AssertionFeatures(),
    )
