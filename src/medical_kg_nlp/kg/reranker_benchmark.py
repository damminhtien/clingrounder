"""Leakage-checked benchmark for graph evidence as a reranker feature."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.kg.ports import KnowledgeGraphRepositoryPort
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.graph_evidence import GraphContextConcept, GraphEvidenceReranker
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.retrieval.adapters import ExactRetrieverAdapter, FTSRetrieverAdapter
from medical_kg_nlp.retrieval.pipeline import RetrievalPipeline
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository
from medical_kg_nlp.utils.hashing import sha256_file

__all__ = ["benchmark_graph_candidate_reranking"]


@dataclass(frozen=True)
class _LinkedAnnotation:
    annotation_id: str
    document_id: str
    text: str
    span: tuple[int, int]
    code: str


@dataclass(frozen=True)
class _BenchmarkQuery:
    annotation: _LinkedAnnotation
    split: str
    neighbors: tuple[_LinkedAnnotation, ...]


def benchmark_graph_candidate_reranking(
    graph_repository: KnowledgeGraphRepositoryPort,
    terminology_repository: TerminologyRepository,
    *,
    documents_path: str | Path,
    annotations_path: str | Path,
    graph_evidence_path: str | Path,
    calibration_split: str = "dev",
    evaluation_split: str = "test",
    graph_source_splits: tuple[str, ...] = ("train",),
    document_prefix: str = "codiesp:",
    source_label: str = "DIAGNOSTICO",
    context_mode: str = "oracle",
    relation_types: tuple[str, ...] = ("CO_OCCURS_WITH",),
    min_support: int = 2,
    candidate_limit: int = 20,
    max_bonus_grid: tuple[float, ...] = (0.0, 0.01, 0.02, 0.04, 0.08),
    max_errors: int = 50,
) -> dict[str, Any]:
    """Calibrate on one split and evaluate once on a disjoint held-out split.

    Oracle mode measures feature potential. Predicted mode uses only exact-unique
    first-pass links from neighboring gold mentions, so no neighboring gold code is
    passed into the reranker.
    """

    if calibration_split == evaluation_split:
        raise ValueError("Calibration and evaluation splits must differ")
    if not graph_source_splits:
        raise ValueError("At least one graph source split is required")
    if {calibration_split, evaluation_split} & set(graph_source_splits):
        raise ValueError("Graph source splits must be disjoint from benchmark splits")
    if candidate_limit < 1 or min_support < 1 or max_errors < 0:
        raise ValueError("Candidate/support limits must be positive")
    if context_mode not in {"oracle", "predicted_exact_unique"}:
        raise ValueError("context_mode must be oracle or predicted_exact_unique")
    if not max_bonus_grid or any(not 0.0 <= value <= 1.0 for value in max_bonus_grid):
        raise ValueError("max_bonus_grid must contain values in [0, 1]")

    documents, split_by_document = _load_documents(
        Path(documents_path),
        document_prefix=document_prefix,
        selected_splits={calibration_split, evaluation_split},
    )
    evidence_documents = _validate_graph_evidence_splits(
        Path(graph_evidence_path),
        split_by_document,
        allowed_splits=set(graph_source_splits),
    )
    queries = _load_queries(
        Path(annotations_path),
        documents,
        split_by_document,
        source_label=source_label,
    )
    retrieval = RetrievalPipeline(
        terminology_repository,
        (
            ExactRetrieverAdapter(terminology_repository),
            FTSRetrieverAdapter(terminology_repository),
        ),
        max_candidates=candidate_limit,
    )
    candidate_cache: dict[str, tuple[Candidate, ...]] = {}
    calibration_queries = tuple(query for query in queries if query.split == calibration_split)
    evaluation_queries = tuple(query for query in queries if query.split == evaluation_split)
    calibration_variants = {
        _bonus_key(bonus): _evaluate_queries(
            calibration_queries,
            retrieval,
            graph_repository,
            candidate_cache,
            context_mode=context_mode,
            relation_types=relation_types,
            min_support=min_support,
            max_bonus=bonus,
            max_errors=max_errors,
        )
        for bonus in sorted(set(max_bonus_grid))
    }
    selected_bonus = _select_bonus(calibration_variants)
    evaluation = _evaluate_queries(
        evaluation_queries,
        retrieval,
        graph_repository,
        candidate_cache,
        context_mode=context_mode,
        relation_types=relation_types,
        min_support=min_support,
        max_bonus=selected_bonus,
        max_errors=max_errors,
    )
    return {
        "schema_version": "graph-evidence-reranker-benchmark.v1",
        "semantic_contract": _semantic_contract(context_mode),
        "leakage_contract": {
            "graph_source_splits": list(graph_source_splits),
            "calibration_split": calibration_split,
            "evaluation_split": evaluation_split,
            "document_backed_graph_evidence_count": evidence_documents,
            "status": "passed",
        },
        "inputs": {
            "documents": _file_identity(Path(documents_path)),
            "annotations": _file_identity(Path(annotations_path)),
            "graph_evidence": _file_identity(Path(graph_evidence_path)),
        },
        "graph_metadata": dict(getattr(graph_repository, "metadata", {})),
        "retrieval": {
            "candidate_limit": candidate_limit,
            "sources": list(retrieval.retrieval_sources),
        },
        "feature": {
            "context_mode": context_mode,
            "relation_types": list(relation_types),
            "min_support": min_support,
            "max_bonus_grid": sorted(set(max_bonus_grid)),
            "selected_max_bonus": selected_bonus,
        },
        "query_counts": {
            calibration_split: len(calibration_queries),
            evaluation_split: len(evaluation_queries),
        },
        "calibration": {
            "split": calibration_split,
            "variants": calibration_variants,
            "selected_max_bonus": selected_bonus,
        },
        "evaluation": {
            "split": evaluation_split,
            **evaluation,
        },
    }


def _load_documents(
    path: Path,
    *,
    document_prefix: str,
    selected_splits: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    documents: dict[str, str] = {}
    split_by_document: dict[str, str] = {}
    for raw in _iter_jsonl(path):
        document_id = str(raw.get("document_id", ""))
        if not document_id.startswith(document_prefix):
            continue
        metadata = raw.get("metadata")
        split = str(metadata.get("corpus_split", "")) if isinstance(metadata, dict) else ""
        if not split:
            raise ValueError(f"Document {document_id} has no corpus_split")
        split_by_document[document_id] = split
        if split in selected_splits:
            documents[document_id] = str(raw.get("text", ""))
    return documents, split_by_document


def _validate_graph_evidence_splits(
    path: Path,
    split_by_document: dict[str, str],
    *,
    allowed_splits: set[str],
) -> int:
    count = 0
    for raw in _iter_jsonl(path):
        document_id = raw.get("document_id")
        if document_id is None:
            continue
        split = split_by_document.get(str(document_id))
        if split is None:
            raise ValueError(f"Graph evidence references unknown document {document_id}")
        if split not in allowed_splits:
            raise ValueError(
                f"Graph evidence document {document_id} belongs to forbidden split {split}"
            )
        count += 1
    return count


def _load_queries(
    path: Path,
    documents: dict[str, str],
    split_by_document: dict[str, str],
    *,
    source_label: str,
) -> tuple[_BenchmarkQuery, ...]:
    annotations_by_document: dict[str, list[_LinkedAnnotation]] = defaultdict(list)
    for raw in _iter_jsonl(path):
        document_id = str(raw.get("document_id", ""))
        if document_id not in documents or raw.get("source_label") != source_label:
            continue
        annotation = _linked_annotation(raw, documents[document_id])
        if annotation is not None:
            annotations_by_document[document_id].append(annotation)

    queries: list[_BenchmarkQuery] = []
    for document_id, document_annotations in sorted(annotations_by_document.items()):
        text = documents[document_id]
        sentences = split_sentences(text)
        sentence_annotations: dict[int, list[_LinkedAnnotation]] = defaultdict(list)
        for annotation in document_annotations:
            sentence_index = _containing_sentence(sentences, annotation.span)
            if sentence_index is not None:
                sentence_annotations[sentence_index].append(annotation)
        for sentence_index, values in sorted(sentence_annotations.items()):
            del sentence_index
            for target in values:
                queries.append(
                    _BenchmarkQuery(
                        annotation=target,
                        split=split_by_document[document_id],
                        neighbors=tuple(
                            item
                            for item in values
                            if item.annotation_id != target.annotation_id
                        ),
                    )
                )
    return tuple(queries)


def _linked_annotation(raw: dict[str, Any], source_text: str) -> _LinkedAnnotation | None:
    if raw.get("entity_type") != EntityType.DISEASE.value:
        return None
    metadata = raw.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("discontinuous") == "true":
            return None
        if metadata.get("source_text_match") == "false":
            return None
    concepts = raw.get("concepts")
    if not isinstance(concepts, list) or len(concepts) != 1:
        return None
    concept = concepts[0]
    if not isinstance(concept, dict) or concept.get("code_system") != CodeSystem.ICD10.value:
        return None
    span = raw.get("span")
    if not isinstance(span, list) or len(span) != 2:
        return None
    start, end = int(span[0]), int(span[1])
    text = str(raw.get("text", ""))
    # INVARIANT: benchmark labels must still address the immutable source text.
    if start < 0 or end <= start or source_text[start:end] != text:
        raise ValueError(f"Annotation {raw.get('annotation_id')} has invalid raw offsets")
    return _LinkedAnnotation(
        annotation_id=str(raw["annotation_id"]),
        document_id=str(raw["document_id"]),
        text=text,
        span=(start, end),
        code=str(concept["code"]),
    )


def _containing_sentence(sentences: list[Any], span: tuple[int, int]) -> int | None:
    for index, sentence in enumerate(sentences):
        if sentence.span[0] <= span[0] and span[1] <= sentence.span[1]:
            return index
    return None


def _evaluate_queries(
    queries: tuple[_BenchmarkQuery, ...],
    retrieval: RetrievalPipeline,
    graph_repository: KnowledgeGraphRepositoryPort,
    candidate_cache: dict[str, tuple[Candidate, ...]],
    *,
    context_mode: str,
    relation_types: tuple[str, ...],
    min_support: int,
    max_bonus: float,
    max_errors: int,
) -> dict[str, Any]:
    reranker = GraphEvidenceReranker(
        graph_repository,
        relation_types=relation_types,
        min_support=min_support,
        max_bonus=max_bonus,
    )
    rows: list[dict[str, Any]] = []
    for query in queries:
        mention = query.annotation.text
        cached = _retrieve_candidates(mention, retrieval, candidate_cache)
        context, anchor_count, correct_anchor_count = _resolve_context(
            query,
            context_mode=context_mode,
            retrieval=retrieval,
            candidate_cache=candidate_cache,
        )
        baseline_candidates = list(cached)
        reranked_candidates = reranker.rerank(
            baseline_candidates,
            mention=mention,
            context_concepts=context,
        )
        rows.append(
            {
                "annotation_id": query.annotation.annotation_id,
                "document_id": query.annotation.document_id,
                "mention": mention,
                "expected_code": query.annotation.code,
                "neighbor_mention_count": len(query.neighbors),
                "context_count": len(context),
                "context_anchor_count": anchor_count,
                "correct_context_anchor_count": correct_anchor_count,
                "baseline_rank": _candidate_rank(
                    baseline_candidates, query.annotation.code
                ),
                "reranked_rank": _candidate_rank(
                    reranked_candidates, query.annotation.code
                ),
                "baseline_top1": _top_code(baseline_candidates),
                "reranked_top1": _top_code(reranked_candidates),
                "graph_feature_used": any(
                    "graph_reranker" in candidate.sources
                    for candidate in reranked_candidates
                ),
            }
        )
    baseline_metrics = _rank_metrics(rows, "baseline_rank")
    reranked_metrics = _rank_metrics(rows, "reranked_rank")
    improved = sum(_rank_order(row["reranked_rank"]) < _rank_order(row["baseline_rank"]) for row in rows)
    worsened = sum(_rank_order(row["reranked_rank"]) > _rank_order(row["baseline_rank"]) for row in rows)
    changed_top1 = sum(row["baseline_top1"] != row["reranked_top1"] for row in rows)
    errors = [
        row
        for row in rows
        if row["reranked_rank"] != 1
    ][:max_errors]
    return {
        "max_bonus": max_bonus,
        "query_count": len(rows),
        "queries_with_context": sum(row["context_count"] > 0 for row in rows),
        "queries_with_graph_feature": sum(row["graph_feature_used"] for row in rows),
        "context_anchors": {
            "mode": context_mode,
            "emitted": sum(row["context_anchor_count"] for row in rows),
            "correct": sum(row["correct_context_anchor_count"] for row in rows),
            "precision": _rate(
                sum(row["correct_context_anchor_count"] for row in rows),
                sum(row["context_anchor_count"] for row in rows),
            ),
        },
        "baseline": baseline_metrics,
        "reranked": reranked_metrics,
        "delta": {
            key: round(reranked_metrics[key] - baseline_metrics[key], 8)
            for key in ("accuracy_at_1", "mrr", "recall_at_5", "recall_at_10", "recall_at_20")
        },
        "rank_changes": {
            "improved": improved,
            "worsened": worsened,
            "unchanged": len(rows) - improved - worsened,
            "changed_top1": changed_top1,
        },
        "sample_errors": errors,
    }


def _rank_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    total = len(rows)
    ranks = [row[key] for row in rows]
    return {
        "accuracy_at_1": _rate(sum(rank == 1 for rank in ranks), total),
        "mrr": round(sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / total, 8) if total else 0.0,
        "recall_at_5": _rate(sum(rank is not None and rank <= 5 for rank in ranks), total),
        "recall_at_10": _rate(sum(rank is not None and rank <= 10 for rank in ranks), total),
        "recall_at_20": _rate(sum(rank is not None and rank <= 20 for rank in ranks), total),
    }


def _resolve_context(
    query: _BenchmarkQuery,
    *,
    context_mode: str,
    retrieval: RetrievalPipeline,
    candidate_cache: dict[str, tuple[Candidate, ...]],
) -> tuple[tuple[GraphContextConcept, ...], int, int]:
    if context_mode == "oracle":
        codes = sorted(
            {
                neighbor.code
                for neighbor in query.neighbors
                if neighbor.code != query.annotation.code
            }
        )
        context = tuple(
            GraphContextConcept(CodeSystem.ICD10, code) for code in codes
        )
        return context, len(context), len(context)

    anchors: dict[str, tuple[str, bool]] = {}
    for neighbor in query.neighbors:
        candidates = _retrieve_candidates(neighbor.text, retrieval, candidate_cache)
        anchor = _exact_unique_anchor(candidates)
        if anchor is None or anchor.code is None:
            continue
        key = f"{anchor.code_system.value}:{anchor.code}"
        is_correct = anchor.code == neighbor.code
        current = anchors.get(key)
        anchors[key] = (
            anchor.code,
            is_correct if current is None else current[1] or is_correct,
        )
    context = tuple(
        GraphContextConcept(CodeSystem.ICD10, code)
        for code in sorted(value[0] for value in anchors.values())
    )
    return context, len(context), sum(value[1] for value in anchors.values())


def _retrieve_candidates(
    mention: str,
    retrieval: RetrievalPipeline,
    cache: dict[str, tuple[Candidate, ...]],
) -> tuple[Candidate, ...]:
    cached = cache.get(mention)
    if cached is None:
        cached = tuple(retrieval.retrieve(mention, EntityType.DISEASE))
        cache[mention] = cached
    return cached


def _exact_unique_anchor(candidates: tuple[Candidate, ...]) -> Candidate | None:
    exact = [
        candidate
        for candidate in candidates
        if candidate.code is not None
        and candidate.code_system == CodeSystem.ICD10
        and "exact" in candidate.sources
    ]
    keys = {(candidate.code_system, candidate.code) for candidate in exact}
    return exact[0] if len(keys) == 1 else None


def _semantic_contract(context_mode: str) -> str:
    if context_mode == "oracle":
        return "oracle_linked_context_upper_bound_not_production_performance"
    return "gold_mentions_with_predicted_exact_unique_context_links"


def _select_bonus(variants: dict[str, dict[str, Any]]) -> float:
    selected_key = max(
        variants,
        key=lambda key: (
            variants[key]["reranked"]["mrr"],
            variants[key]["reranked"]["accuracy_at_1"],
            variants[key]["reranked"]["recall_at_5"],
            -float(key),
        ),
    )
    return float(selected_key)


def _candidate_rank(candidates: list[Candidate], expected_code: str) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if candidate.code_system == CodeSystem.ICD10 and candidate.code == expected_code:
            return rank
    return None


def _top_code(candidates: list[Candidate]) -> str | None:
    return candidates[0].code if candidates else None


def _rank_order(rank: object) -> float:
    return float(rank) if isinstance(rank, int) else float("inf")


def _bonus_key(value: float) -> str:
    return f"{value:.8g}"


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _file_identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield raw
