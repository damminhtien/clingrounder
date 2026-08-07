from __future__ import annotations
from clingrounder.schema.annotation import RelationAnnotation
from clingrounder.evaluation.span_metrics import PRF, _prf


def relation_f1(gold: list[RelationAnnotation], pred: list[RelationAnnotation]) -> PRF:
    gold_keys = {(relation.head, relation.tail, relation.type) for relation in gold}
    pred_keys = {(relation.head, relation.tail, relation.type) for relation in pred}
    return _prf(len(gold_keys & pred_keys), len(pred_keys - gold_keys), len(gold_keys - pred_keys))

