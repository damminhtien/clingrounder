from __future__ import annotations
from collections import Counter

from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus


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


def confusion_matrix(gold: list[EntityAnnotation], pred: list[EntityAnnotation]) -> dict[str, dict[str, int]]:
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
        false_positive = sum(counts[(other.value, label)] for other in AssertionStatus if other.value != label)
        false_negative = sum(counts[(label, other.value)] for other in AssertionStatus if other.value != label)
        support = true_positive + false_negative
        if support == 0:
            continue
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support
        label_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(label_scores) / len(label_scores) if label_scores else 0.0
