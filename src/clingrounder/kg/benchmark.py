"""Offline benchmarks for exact alias coverage in a compiled knowledge graph."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from clingrounder.kg.knowledge_schema import KnowledgeNode
from clingrounder.kg.ports import KnowledgeGraphRepositoryPort

__all__ = ["benchmark_graph_aliases", "benchmark_graph_relations"]


def benchmark_graph_relations(
    repository: KnowledgeGraphRepositoryPort,
    edge_path: str | Path,
    *,
    relation_type: str,
    workers: int = 8,
    repeats: int = 3,
    limit: int = 100,
    max_misses: int = 50,
) -> dict[str, Any]:
    """Verify indexed edge coverage, deterministic reads, and traversal latency.

    This is an index-consistency benchmark, not a clinical relation-quality metric.  It
    checks that an immutable edge artifact is queryable through the read-only repository.
    """

    if not relation_type.strip():
        raise ValueError("relation_type must be non-empty")
    if workers < 1 or repeats < 1 or limit < 1 or max_misses < 0:
        raise ValueError("workers, repeats, and limit must be positive")
    expected_by_head = _expected_relation_neighbors(Path(edge_path), relation_type)
    query_limit = max(limit, max((len(values) for values in expected_by_head.values()), default=0))

    def query(head_node_id: str) -> tuple[str, tuple[str, ...], float]:
        started = time.perf_counter()
        neighbors = repository.neighbors(
            head_node_id,
            direction="outgoing",
            relation_types=(relation_type,),
            min_support=1,
            limit=query_limit,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return (
            head_node_id,
            tuple(sorted(neighbor.node.node_id for neighbor in neighbors)),
            elapsed_ms,
        )

    heads = sorted(expected_by_head)
    tasks = [head for _ in range(repeats) for head in heads]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(query, tasks))
    elapsed_ms = (time.perf_counter() - started) * 1000

    observed_by_head: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    latencies = []
    for head, observed, latency in results:
        observed_by_head[head].append(observed)
        latencies.append(latency)
    deterministic = all(
        len(set(observations)) == 1 for observations in observed_by_head.values()
    )
    covered = 0
    missing: list[dict[str, str]] = []
    for head in heads:
        observed_nodes = (
            set(observed_by_head[head][0]) if observed_by_head[head] else set()
        )
        for tail in sorted(expected_by_head[head]):
            if tail in observed_nodes:
                covered += 1
            elif len(missing) < max_misses:
                missing.append({"head_node_id": head, "tail_node_id": tail})
    expected_edge_count = sum(len(values) for values in expected_by_head.values())
    return {
        "schema_version": "knowledge-graph-relation-benchmark.v1",
        "index_metadata": dict(getattr(repository, "metadata", {})),
        "edge_source": str(Path(edge_path)),
        "relation_type": relation_type,
        "expected_edge_count": expected_edge_count,
        "covered_edge_count": covered,
        "coverage_rate": _rate(covered, expected_edge_count),
        "head_node_count": len(heads),
        "query_count": len(results),
        "query_limit": query_limit,
        "workers": workers,
        "repeats": repeats,
        "deterministic": deterministic,
        "elapsed_ms": round(elapsed_ms, 3),
        "queries_per_second": (
            round(len(results) / (elapsed_ms / 1000), 3) if elapsed_ms else 0.0
        ),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 6) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.50), 6),
            "p95": round(_percentile(latencies, 0.95), 6),
            "max": round(max(latencies), 6) if latencies else 0.0,
        },
        "sample_misses": missing,
        "semantic_contract": "index_consistency_not_clinical_relation_quality",
    }


def benchmark_graph_aliases(
    repository: KnowledgeGraphRepositoryPort,
    overlay_paths: tuple[str | Path, ...],
    *,
    limit: int = 5,
    max_misses: int = 50,
) -> dict[str, Any]:
    """Measure target coverage and exact/toneless rank for alias overlays.

    This benchmark deliberately calls the exact-only graph path. It therefore measures
    whether a mined alias is safely usable for candidate generation, rather than allowing
    FTS to hide an alias normalization or provenance defect.
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    if max_misses < 0:
        raise ValueError("max_misses must be non-negative")
    sources: dict[str, dict[str, Any]] = {}
    for raw_path in overlay_paths:
        path = Path(raw_path)
        sources[str(path)] = _benchmark_one(
            repository,
            path,
            limit=limit,
            max_misses=max_misses,
        )
    return {
        "schema_version": "knowledge-graph-alias-benchmark.v1",
        "index_metadata": dict(getattr(repository, "metadata", {})),
        "limit": limit,
        "sources": sources,
    }


def _benchmark_one(
    repository: KnowledgeGraphRepositoryPort,
    path: Path,
    *,
    limit: int,
    max_misses: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = 0
    target_rows = 0
    covered_targets = 0
    unknown_targets = 0
    top1 = 0
    topk = 0
    ambiguous = 0
    misses: list[dict[str, str | None]] = []
    expected_cache: dict[tuple[str, str], KnowledgeNode | None] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows += 1
            system, code = _target_code(raw)
            if system is None or code is None:
                continue
            target_rows += 1
            key = (system, code)
            if key not in expected_cache:
                expected_cache[key] = repository.get_by_code(system, code)
            expected = expected_cache[key]
            if expected is None:
                unknown_targets += 1
                continue
            covered_targets += 1
            semantic_type = raw.get("semantic_type")
            entity_type = semantic_type if isinstance(semantic_type, str) else None
            results = repository.search_nodes(
                str(raw.get("alias", "")),
                entity_type=entity_type,
                code_system=system,
                limit=limit,
                exact_only=True,
            )
            if len(results) > 1:
                ambiguous += 1
            result_ids = [node.node_id for node in results]
            expected_id = expected.node_id
            if result_ids and result_ids[0] == expected_id:
                top1 += 1
            if expected_id in result_ids:
                topk += 1
            elif len(misses) < max_misses:
                misses.append(
                    {
                        "line": str(line_number),
                        "alias": str(raw.get("alias", "")),
                        "expected": f"{system}:{code}",
                        "returned": (
                            None
                            if not results
                            else f"{results[0].code_system}:{results[0].code}"
                        ),
                    }
                )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "rows": rows,
        "target_rows": target_rows,
        "covered_targets": covered_targets,
        "unknown_targets": unknown_targets,
        "top1": top1,
        "top1_rate": _rate(top1, covered_targets),
        "topk": topk,
        "topk_rate": _rate(topk, covered_targets),
        "ambiguous_queries": ambiguous,
        "elapsed_ms": round(elapsed_ms, 3),
        "rows_per_second": round(rows / (elapsed_ms / 1000), 3) if elapsed_ms else 0.0,
        "sample_misses": misses,
    }


def _target_code(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    target = raw.get("target_concept_id")
    if isinstance(target, str) and ":" in target:
        prefix, code = target.split(":", 1)
        if prefix == "ICD10":
            return "ICD-10", code
        if prefix == "RXNORM":
            return "RxNorm", code
    raw_system = raw.get("code_system")
    raw_code = raw.get("code")
    if not isinstance(raw_system, str) or not isinstance(raw_code, str):
        return None, None
    return raw_system, raw_code


def _expected_relation_neighbors(
    path: Path,
    relation_type: str,
) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            if raw.get("relation_type") != relation_type:
                continue
            head = raw.get("head_node_id")
            tail = raw.get("tail_node_id")
            if not isinstance(head, str) or not head.strip():
                raise ValueError(f"{path}:{line_number}: invalid head_node_id")
            if not isinstance(tail, str) or not tail.strip():
                raise ValueError(f"{path}:{line_number}: invalid tail_node_id")
            expected[head].add(tail)
    return dict(expected)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
