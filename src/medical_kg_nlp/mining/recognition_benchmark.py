"""Measure whether mined recognition dictionaries improve exact entity extraction."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, median
from time import perf_counter
from typing import Any

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.merge import merge_concept_entries
from medical_kg_nlp.mining.records import AnnotationProposal, MinedDocument
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.schema.types import EntityType

__all__ = ["benchmark_recognition_dictionary"]


def benchmark_recognition_dictionary(
    documents: Sequence[MinedDocument],
    gold_annotations: Sequence[AnnotationProposal],
    baseline_store: DictionaryStore,
    additional_store: DictionaryStore,
    *,
    entity_types: Sequence[EntityType],
) -> dict[str, Any]:
    """Compare baseline and enriched dictionary matchers on immutable source offsets."""

    if not documents or not entity_types:
        raise ValueError("Recognition benchmark requires documents and entity types")
    document_ids = {document.document_id for document in documents}
    selected_types = set(entity_types)
    gold: set[tuple[str, int, int, EntityType]] = set()
    for annotation in gold_annotations:
        if annotation.document_id not in document_ids:
            raise ValueError(
                f"Gold annotation references unknown document {annotation.document_id!r}"
            )
        try:
            entity_type = EntityType(annotation.entity_type)
        except ValueError:
            continue
        if entity_type in selected_types:
            gold.add(
                (
                    annotation.document_id,
                    annotation.span[0],
                    annotation.span[1],
                    entity_type,
                )
            )

    enriched_store = DictionaryStore(
        merge_concept_entries((*baseline_store.entries, *additional_store.entries))
    )
    baseline = _run_extractor(documents, gold, baseline_store, selected_types)
    enriched = _run_extractor(documents, gold, enriched_store, selected_types)
    return {
        "schema_version": "mined-recognition-benchmark.v1",
        "document_count": len(documents),
        "gold_entity_count": len(gold),
        "entity_types": sorted(entity_type.value for entity_type in selected_types),
        "baseline": baseline,
        "enriched": enriched,
        "delta": {
            "exact_precision": enriched["metrics"]["precision"]
            - baseline["metrics"]["precision"],
            "exact_recall": enriched["metrics"]["recall"]
            - baseline["metrics"]["recall"],
            "exact_f1": enriched["metrics"]["f1"] - baseline["metrics"]["f1"],
            "true_positive_count": enriched["metrics"]["true_positive_count"]
            - baseline["metrics"]["true_positive_count"],
            "false_positive_count": enriched["metrics"]["false_positive_count"]
            - baseline["metrics"]["false_positive_count"],
        },
    }


def _run_extractor(
    documents: Sequence[MinedDocument],
    gold: set[tuple[str, int, int, EntityType]],
    store: DictionaryStore,
    selected_types: set[EntityType],
) -> dict[str, Any]:
    build_started = perf_counter()
    extractor = RuleBasedNER(store)
    build_ms = (perf_counter() - build_started) * 1000.0
    predicted: set[tuple[str, int, int, EntityType]] = set()
    latencies: list[float] = []
    for document in documents:
        started = perf_counter()
        entities = extractor.extract(document.text)
        latencies.append((perf_counter() - started) * 1000.0)
        for entity in entities:
            if entity.type not in selected_types:
                continue
            # INVARIANT: benchmarked matches must still address the immutable mined document.
            entity.validate_offsets(document.text)
            predicted.add(
                (
                    document.document_id,
                    entity.span[0],
                    entity.span[1],
                    entity.type,
                )
            )
    metrics = _exact_metrics(gold, predicted)
    return {
        "dictionary_concept_count": len(store.entries),
        "dictionary_alias_count": len(store.aliases_for_ner()),
        "prediction_count": len(predicted),
        "metrics": metrics,
        "entity_types": _type_metrics(gold, predicted),
        "runtime_ms": {
            "matcher_build": build_ms,
            "extraction_total": sum(latencies),
            "per_document_mean": fmean(latencies),
            "per_document_p50": median(latencies),
            "per_document_p95": sorted(latencies)[
                min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))
            ],
            "per_document_max": max(latencies),
        },
    }


def _exact_metrics(
    gold: set[tuple[str, int, int, EntityType]],
    predicted: set[tuple[str, int, int, EntityType]],
) -> dict[str, float | int]:
    true_positive = len(gold & predicted)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    precision = true_positive / (true_positive + false_positive) if predicted else 0.0
    recall = true_positive / (true_positive + false_negative) if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
    }


def _type_metrics(
    gold: set[tuple[str, int, int, EntityType]],
    predicted: set[tuple[str, int, int, EntityType]],
) -> dict[str, dict[str, float | int]]:
    types = sorted({row[3] for row in gold | predicted}, key=lambda value: value.value)
    return {
        entity_type.value: _exact_metrics(
            {row for row in gold if row[3] == entity_type},
            {row for row in predicted if row[3] == entity_type},
        )
        for entity_type in types
    }
