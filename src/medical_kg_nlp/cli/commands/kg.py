"""Build and inspect persistent medical knowledge graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical_kg_nlp.kg import (
    SQLiteKnowledgeGraphRepository,
    build_knowledge_graph_index,
)
from medical_kg_nlp.kg.benchmark import (
    benchmark_graph_aliases,
    benchmark_graph_relations,
)

__all__ = [
    "benchmark_graph_aliases_command",
    "benchmark_graph_relations_command",
    "build_graph_index",
    "inspect_graph_index",
]


def benchmark_graph_aliases_command(args: argparse.Namespace) -> int:
    """Run a compact-polling alias benchmark and persist the full report."""

    repository = SQLiteKnowledgeGraphRepository(args.index)
    try:
        report = benchmark_graph_aliases(
            repository,
            tuple(args.alias_overlay),
            limit=args.limit,
            max_misses=args.max_misses,
        )
    finally:
        repository.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": report["schema_version"],
        "sources": {
            path: {
                "covered_targets": values["covered_targets"],
                "top1_rate": values["top1_rate"],
                "topk_rate": values["topk_rate"],
                "ambiguous_queries": values["ambiguous_queries"],
                "elapsed_ms": values["elapsed_ms"],
            }
            for path, values in report["sources"].items()
        },
        "report": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def benchmark_graph_relations_command(args: argparse.Namespace) -> int:
    """Benchmark immutable relation-edge traversal and concurrent read stability."""

    repository = SQLiteKnowledgeGraphRepository(args.index)
    try:
        report = benchmark_graph_relations(
            repository,
            args.edges,
            relation_type=args.relation_type,
            workers=args.workers,
            repeats=args.repeats,
            limit=args.limit,
            max_misses=args.max_misses,
        )
    finally:
        repository.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "coverage_rate": report["coverage_rate"],
                "deterministic": report["deterministic"],
                "expected_edge_count": report["expected_edge_count"],
                "latency_ms": report["latency_ms"],
                "queries_per_second": report["queries_per_second"],
                "report": str(output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_graph_index(args: argparse.Namespace) -> int:
    """Build a content-addressed graph and persist its reproducibility manifest."""

    manifest = build_knowledge_graph_index(
        args.nodes,
        args.edges,
        args.evidence,
        output_path=args.output,
        cache_dir=args.cache_dir,
    )
    payload = manifest.to_json()
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.manifest_output is not None:
        output = Path(args.manifest_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


def inspect_graph_index(args: argparse.Namespace) -> int:
    """Query nodes, neighbors, hierarchy, or evidence from a pinned graph."""

    if args.code is not None and args.code_system is None:
        raise ValueError("--code requires --code-system")
    repository = SQLiteKnowledgeGraphRepository(
        args.index,
        expected_nodes_path=args.nodes,
        expected_edges_path=args.edges,
        expected_evidence_path=args.evidence,
    )
    try:
        payload: dict[str, object] = {"metadata": repository.metadata}
        if args.query:
            payload["search_results"] = [
                node.to_dict()
                for node in repository.search_nodes(
                    args.query,
                    entity_type=args.entity_type,
                    code_system=args.code_system,
                    limit=args.limit,
                )
            ]
        if args.code_system and args.code:
            node = repository.get_by_code(args.code_system, args.code)
            payload["code_result"] = None if node is None else node.to_dict()
        if args.node_id:
            node = repository.get_node(args.node_id)
            payload["node"] = None if node is None else node.to_dict()
            payload["neighbors"] = [
                {
                    "direction": neighbor.direction,
                    "edge": neighbor.edge.to_dict(),
                    "node": neighbor.node.to_dict(),
                }
                for neighbor in repository.neighbors(
                    args.node_id,
                    direction=args.direction,
                    relation_types=tuple(args.relation_type),
                    min_support=args.min_support,
                    limit=args.limit,
                )
            ]
            if args.ancestors:
                payload["ancestors"] = [
                    {"distance": distance, "node": ancestor.to_dict()}
                    for ancestor, distance in repository.ancestors(
                        args.node_id,
                        max_depth=args.max_depth,
                    )
                ]
        if args.edge_id:
            payload["evidence"] = [
                item.to_dict() for item in repository.evidence(args.edge_id, limit=args.limit)
            ]
    finally:
        repository.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
