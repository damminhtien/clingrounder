"""Metrics for primary and independent clinical assertion attributes.

Primary-label metrics preserve the original baseline contract. Attribute metrics
separate negation, history, experiencer, and uncertainty so one dominant label
cannot hide errors in another assertion dimension.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from clingrounder.schema.annotation import EntityAnnotation
from clingrounder.schema.types import AssertionStatus, EntityType

__all__ = [
    "assertion_attribute_metrics",
    "confusion_counts",
    "confusion_matrix",
    "context_accuracy",
    "context_macro_f1",
]

_ASSERTION_ATTRIBUTES: tuple[tuple[str, AssertionStatus], ...] = (
    ("negated", AssertionStatus.NEGATED),
    ("historical", AssertionStatus.HISTORICAL),
    ("family", AssertionStatus.FAMILY),
    ("possible", AssertionStatus.POSSIBLE),
    ("conditional", AssertionStatus.CONDITIONAL),
    ("planned", AssertionStatus.PLANNED),
    ("resolved", AssertionStatus.RESOLVED),
)


def context_accuracy(gold: list[EntityAnnotation], pred: list[EntityAnnotation]) -> float:
    pred_by_span = {entity.span: entity for entity in pred}
    total = 0
    correct = 0
    for gold_entity in gold:
        prediction = pred_by_span.get(gold_entity.span)
        if prediction is None:
            continue
        total += 1
        if prediction.assertion == gold_entity.assertion:
            correct += 1
    return correct / total if total else 0.0


def confusion_counts(
    gold: list[EntityAnnotation],
    pred: list[EntityAnnotation],
) -> Counter[tuple[str, str]]:
    pred_by_span = {entity.span: entity for entity in pred}
    counts: Counter[tuple[str, str]] = Counter()
    for gold_entity in gold:
        prediction = pred_by_span.get(gold_entity.span)
        if prediction is not None:
            counts[(gold_entity.assertion.value, prediction.assertion.value)] += 1
    return counts


def confusion_matrix(
    gold: list[EntityAnnotation],
    pred: list[EntityAnnotation],
) -> dict[str, dict[str, int]]:
    counts = confusion_counts(gold, pred)
    labels = [status.value for status in AssertionStatus]
    return {
        gold_label: {
            pred_label: counts[(gold_label, pred_label)]
            for pred_label in labels
            if counts[(gold_label, pred_label)] > 0
        }
        for gold_label in labels
        if any(counts[(gold_label, pred_label)] > 0 for pred_label in labels)
    }


def context_macro_f1(gold: list[EntityAnnotation], pred: list[EntityAnnotation]) -> float:
    counts = confusion_counts(gold, pred)
    label_scores: list[float] = []
    for status in AssertionStatus:
        label = status.value
        true_positive = counts[(label, label)]
        false_positive = sum(
            counts[(other.value, label)]
            for other in AssertionStatus
            if other.value != label
        )
        false_negative = sum(
            counts[(label, other.value)]
            for other in AssertionStatus
            if other.value != label
        )
        support = true_positive + false_negative
        if support == 0:
            continue
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = true_positive / support
        label_scores.append(_safe_ratio(2 * precision * recall, precision + recall))
    return sum(label_scores) / len(label_scores) if label_scores else 0.0


def assertion_attribute_metrics(
    gold: list[EntityAnnotation],
    pred: list[EntityAnnotation],
) -> dict[str, Any]:
    """Score each assertion attribute on entities with exact span and type.

    NER projection errors are reported separately instead of being counted as
    assertion errors. This makes assertion-only ablations interpretable.
    """

    gold_by_key = {_entity_key(entity): entity for entity in gold}
    pred_by_key = {_entity_key(entity): entity for entity in pred}
    matched_keys = gold_by_key.keys() & pred_by_key.keys()
    attribute_rows: dict[str, dict[str, float | int]] = {}
    active_f1_scores: list[float] = []

    for attribute, status in _ASSERTION_ATTRIBUTES:
        true_positive = false_positive = false_negative = true_negative = 0
        for key in matched_keys:
            gold_positive = status in _effective_statuses(gold_by_key[key])
            pred_positive = status in _effective_statuses(pred_by_key[key])
            if gold_positive and pred_positive:
                true_positive += 1
            elif pred_positive:
                false_positive += 1
            elif gold_positive:
                false_negative += 1
            else:
                true_negative += 1

        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        support = true_positive + false_negative
        if support:
            active_f1_scores.append(f1)
        attribute_rows[attribute] = {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "tn": true_negative,
            "support": support,
            "predicted_positive": true_positive + false_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "matched_entity_count": len(matched_keys),
        "missing_gold_count": len(gold_by_key.keys() - pred_by_key.keys()),
        "spurious_prediction_count": len(pred_by_key.keys() - gold_by_key.keys()),
        "active_attribute_macro_f1": (
            sum(active_f1_scores) / len(active_f1_scores) if active_f1_scores else 0.0
        ),
        "attributes": attribute_rows,
    }


def _entity_key(entity: EntityAnnotation) -> tuple[tuple[int, int], EntityType]:
    return entity.span, entity.type


def _effective_statuses(entity: EntityAnnotation) -> set[AssertionStatus]:
    statuses = set(entity.assertion_features.statuses())
    if entity.assertion not in {AssertionStatus.PRESENT, AssertionStatus.UNKNOWN}:
        statuses.add(entity.assertion)
    return statuses


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return numerator / denominator if denominator else 0.0
