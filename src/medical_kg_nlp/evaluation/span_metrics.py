from __future__ import annotations
from dataclasses import dataclass

from medical_kg_nlp.schema.annotation import EntityAnnotation


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def exact_span_f1(gold: list[EntityAnnotation], pred: list[EntityAnnotation]) -> PRF:
    gold_keys = {(entity.span, entity.type) for entity in gold}
    pred_keys = {(entity.span, entity.type) for entity in pred}
    tp = len(gold_keys & pred_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)
    return _prf(tp, fp, fn)


def overlap_span_f1(gold: list[EntityAnnotation], pred: list[EntityAnnotation]) -> PRF:
    matched_gold: set[int] = set()
    tp = 0
    for prediction in pred:
        for gold_index, gold_entity in enumerate(gold):
            if gold_index in matched_gold or prediction.type != gold_entity.type:
                continue
            if prediction.span[0] < gold_entity.span[1] and gold_entity.span[0] < prediction.span[1]:
                matched_gold.add(gold_index)
                tp += 1
                break
    fp = len(pred) - tp
    fn = len(gold) - tp
    return _prf(tp, fp, fn)


def _prf(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)

