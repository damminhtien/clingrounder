"""Offline benchmarks for exact alias coverage in a compiled knowledge graph."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from medical_kg_nlp.kg.knowledge_schema import KnowledgeNode
from medical_kg_nlp.kg.ports import KnowledgeGraphRepositoryPort

__all__ = ["benchmark_graph_aliases"]


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


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
