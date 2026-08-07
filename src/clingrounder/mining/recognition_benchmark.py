"""Measure whether mined recognition dictionaries improve exact entity extraction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from statistics import fmean, median
from time import perf_counter
from typing import Any

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.dictionaries.merge import merge_concept_entries
from clingrounder.mining.records import AnnotationProposal, MinedDocument
from clingrounder.ner.rule_ner import RuleBasedNER
from clingrounder.schema.types import EntityType
from clingrounder.utils.text import normalize_for_match

__all__ = ["benchmark_recognition_dictionary"]

_SpanKey = tuple[str, int, int, EntityType]


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
    gold: set[_SpanKey] = set()
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
    gold: set[_SpanKey],
    store: DictionaryStore,
    selected_types: set[EntityType],
) -> dict[str, Any]:
    build_started = perf_counter()
    extractor = RuleBasedNER(store)
    build_ms = (perf_counter() - build_started) * 1000.0
    predicted: set[_SpanKey] = set()
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
        "error_analysis": _error_analysis(documents, gold, predicted),
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
    gold: set[_SpanKey],
    predicted: set[_SpanKey],
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
    gold: set[_SpanKey],
    predicted: set[_SpanKey],
) -> dict[str, dict[str, float | int]]:
    types = sorted({row[3] for row in gold | predicted}, key=lambda value: value.value)
    return {
        entity_type.value: _exact_metrics(
            {row for row in gold if row[3] == entity_type},
            {row for row in predicted if row[3] == entity_type},
        )
        for entity_type in types
    }


def _error_analysis(
    documents: Sequence[MinedDocument],
    gold: set[_SpanKey],
    predicted: set[_SpanKey],
) -> dict[str, Any]:
    documents_by_id = {document.document_id: document for document in documents}
    false_positive = predicted - gold
    false_negative = gold - predicted
    return {
        "false_positive": _summarize_errors(
            false_positive,
            reference=gold,
            documents_by_id=documents_by_id,
            overlap_kind="boundary_overlap",
            disjoint_kind="spurious",
        ),
        "false_negative": _summarize_errors(
            false_negative,
            reference=predicted,
            documents_by_id=documents_by_id,
            overlap_kind="boundary_overlap",
            disjoint_kind="missing",
        ),
    }


def _summarize_errors(
    rows: set[_SpanKey],
    *,
    reference: set[_SpanKey],
    documents_by_id: dict[str, MinedDocument],
    overlap_kind: str,
    disjoint_kind: str,
    limit: int = 100,
) -> dict[str, Any]:
    grouped: dict[tuple[str, EntityType, str], list[_SpanKey]] = {}
    for row in sorted(rows):
        document_id, start, end, entity_type = row
        document = documents_by_id[document_id]
        normalized = normalize_for_match(document.text[start:end])
        kind = overlap_kind if _overlaps_reference(row, reference) else disjoint_kind
        grouped.setdefault((normalized, entity_type, kind), []).append(row)

    groups = []
    for (normalized, entity_type, kind), members in sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1].value, item[0][2]),
    )[:limit]:
        examples = []
        for document_id, start, end, _ in members[:3]:
            text = documents_by_id[document_id].text[start:end]
            examples.append(
                {
                    "document_id": document_id,
                    "span": [start, end],
                    "text": text,
                }
            )
        groups.append(
            {
                "normalized_mention": normalized,
                "entity_type": entity_type.value,
                "error_kind": kind,
                "occurrence_count": len(members),
                "document_count": len({row[0] for row in members}),
                "examples": examples,
            }
        )
    kind_counts = Counter(
        overlap_kind if _overlaps_reference(row, reference) else disjoint_kind
        for row in rows
    )
    return {
        "count": len(rows),
        "kind_counts": dict(sorted(kind_counts.items())),
        "top_mentions": groups,
    }


def _overlaps_reference(row: _SpanKey, reference: set[_SpanKey]) -> bool:
    document_id, start, end, entity_type = row
    return any(
        document_id == other_document_id
        and entity_type == other_type
        and start < other_end
        and other_start < end
        for other_document_id, other_start, other_end, other_type in reference
    )
