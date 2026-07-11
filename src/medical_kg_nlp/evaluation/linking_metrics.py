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
        if prediction and any(
            candidate.code == gold_entity.code for candidate in prediction.candidates[:k]
        ):
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


def coverage_accuracy_curve(
    gold: list[EntityAnnotation],
    pred: list[EntityAnnotation],
    thresholds: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95),
) -> list[dict[str, float | int]]:
    pred_by_span = {entity.span: entity for entity in pred}
    eligible = [entity for entity in gold if entity.code is not None]
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        covered = 0
        correct = 0
        for gold_entity in eligible:
            prediction = pred_by_span.get(gold_entity.span)
            if prediction is None or not prediction.candidates:
                continue
            top = prediction.candidates[0]
            if top.code is None or top.score < threshold:
                continue
            covered += 1
            correct += int(top.code == gold_entity.code)
        rows.append(
            {
                "threshold": threshold,
                "eligible": len(eligible),
                "covered": covered,
                "coverage": covered / len(eligible) if eligible else 0.0,
                "accuracy": correct / covered if covered else 0.0,
            }
        )
    return rows
