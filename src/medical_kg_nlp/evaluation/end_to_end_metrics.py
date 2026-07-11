from __future__ import annotations
from typing import Any

from medical_kg_nlp.evaluation.context_metrics import (
    context_accuracy,
    context_macro_f1,
    confusion_matrix,
)
from medical_kg_nlp.evaluation.linking_metrics import (
    accuracy_at_1,
    coverage_accuracy_curve,
    mean_reciprocal_rank,
    recall_at_k,
)
from medical_kg_nlp.evaluation.relation_metrics import relation_f1
from medical_kg_nlp.evaluation.span_metrics import exact_span_f1, overlap_span_f1
from medical_kg_nlp.schema.output import ClinicalPrediction


def evaluate_predictions(
    gold: list[ClinicalPrediction],
    pred: list[ClinicalPrediction],
) -> dict[str, Any]:
    pred_by_doc = {item.document_id: item for item in pred}
    gold_entities = []
    pred_entities = []
    gold_relations = []
    pred_relations = []
    for gold_doc in gold:
        pred_doc = pred_by_doc.get(gold_doc.document_id)
        if pred_doc is None:
            gold_entities.extend(gold_doc.entities)
            gold_relations.extend(gold_doc.relations)
            continue
        gold_entities.extend(gold_doc.entities)
        pred_entities.extend(pred_doc.entities)
        gold_relations.extend(gold_doc.relations)
        pred_relations.extend(pred_doc.relations)

    exact = exact_span_f1(gold_entities, pred_entities)
    overlap = overlap_span_f1(gold_entities, pred_entities)
    rel = relation_f1(gold_relations, pred_relations)
    return {
        "span_exact": exact.__dict__,
        "span_overlap": overlap.__dict__,
        "linking_accuracy_at_1": accuracy_at_1(gold_entities, pred_entities),
        "linking_recall_at_5": recall_at_k(gold_entities, pred_entities, 5),
        "linking_recall_at_10": recall_at_k(gold_entities, pred_entities, 10),
        "linking_recall_at_20": recall_at_k(gold_entities, pred_entities, 20),
        "linking_mrr": mean_reciprocal_rank(gold_entities, pred_entities),
        "linking_coverage_accuracy_curve": coverage_accuracy_curve(gold_entities, pred_entities),
        "context_accuracy": context_accuracy(gold_entities, pred_entities),
        "context_macro_f1": context_macro_f1(gold_entities, pred_entities),
        "context_confusion_matrix": confusion_matrix(gold_entities, pred_entities),
        "relation": rel.__dict__,
    }
