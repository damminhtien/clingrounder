"""Build and inspect persistent medical knowledge graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical_kg_nlp.kg import (
    SQLiteKnowledgeGraphRepository,
    build_knowledge_graph_index,
)

__all__ = ["build_graph_index", "inspect_graph_index"]


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
