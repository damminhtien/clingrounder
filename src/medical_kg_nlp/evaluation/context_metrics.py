from __future__ import annotations
from collections import Counter

from medical_kg_nlp.schema.annotation import EntityAnnotation


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


def confusion_counts(gold: list[EntityAnnotation], pred: list[EntityAnnotation]) -> Counter[tuple[str, str]]:
    pred_by_span = {entity.span: entity for entity in pred}
    counts: Counter[tuple[str, str]] = Counter()
    for gold_entity in gold:
        prediction = pred_by_span.get(gold_entity.span)
        if prediction is not None:
            counts[(gold_entity.assertion.value, prediction.assertion.value)] += 1
    return counts

