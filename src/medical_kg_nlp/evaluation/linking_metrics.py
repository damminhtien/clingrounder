from __future__ import annotations
from medical_kg_nlp.schema.annotation import EntityAnnotation


def accuracy_at_1(gold: list[EntityAnnotation], pred: list[EntityAnnotation]) -> float:
    pred_by_span = {entity.span: entity for entity in pred}
    total = 0
    correct = 0
    for gold_entity in gold:
        if gold_entity.code is None:
            continue
        total += 1
        prediction = pred_by_span.get(gold_entity.span)
        if prediction and prediction.code == gold_entity.code:
            correct += 1
    return correct / total if total else 0.0


def recall_at_k(gold: list[EntityAnnotation], pred: list[EntityAnnotation], k: int) -> float:
    pred_by_span = {entity.span: entity for entity in pred}
    total = 0
    found = 0
    for gold_entity in gold:
        if gold_entity.code is None:
            continue
        total += 1
        prediction = pred_by_span.get(gold_entity.span)
        if prediction and any(candidate.code == gold_entity.code for candidate in prediction.candidates[:k]):
            found += 1
    return found / total if total else 0.0


def mean_reciprocal_rank(gold: list[EntityAnnotation], pred: list[EntityAnnotation]) -> float:
    pred_by_span = {entity.span: entity for entity in pred}
    reciprocal_ranks: list[float] = []
    for gold_entity in gold:
        if gold_entity.code is None:
            continue
        prediction = pred_by_span.get(gold_entity.span)
        rank = 0
        if prediction:
            for index, candidate in enumerate(prediction.candidates, start=1):
                if candidate.code == gold_entity.code:
                    rank = index
                    break
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

