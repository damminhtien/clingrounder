"""Contract tests for the persistent read-only knowledge graph repository."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from medical_kg_nlp.kg.knowledge_schema import (
    KnowledgeEdge,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeKind,
)
from medical_kg_nlp.kg.benchmark import (
    benchmark_graph_aliases,
    benchmark_graph_relations,
)
from medical_kg_nlp.kg.sqlite_builder import build_knowledge_graph_index
from medical_kg_nlp.kg.sqlite_repository import SQLiteKnowledgeGraphRepository
from medical_kg_nlp.cli import main
from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.retrieval.adapters import KnowledgeGraphExactRetrieverAdapter
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match


def _node(
    node_id: str,
    label: str,
    entity_type: str,
    *,
    code_system: str | None = None,
    code: str | None = None,
    aliases: tuple[str, ...] = (),
) -> KnowledgeNode:
    kind = KnowledgeNodeKind.CONCEPT if code is not None else KnowledgeNodeKind.TERM
    return KnowledgeNode(
        node_id=node_id,
        kind=kind,
        label=label,
        normalized_label=normalize_for_match(label),
        entity_type=entity_type,
        code_system=code_system,
        code=code,
        aliases=aliases,
        occurrence_count=2 if code_system in {"NDC", "NCI"} else 0,
        document_count=2 if code_system in {"NDC", "NCI"} else 0,
    )


def _write_graph(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, KnowledgeNode]]:
    nodes = {
        "root": _node("root", "root disease", "DISEASE", code_system="ICD-10", code="I00"),
        "child": _node(
            "child",
            "hypertension",
            "DISEASE",
            code_system="ICD-10",
            code="I10",
            aliases=("cao huyết áp",),
        ),
        "drug": _node("drug", "Drug A", "DRUG", code_system="NDC", code="111"),
        "route": _node("route", "ORAL", "ROUTE", code_system="NCI", code="C1"),
    }
    hierarchy = KnowledgeEdge(
        edge_id="edge-hierarchy",
        head_node_id="child",
        tail_node_id="root",
        relation_type="IS_A",
        support_count=1,
        document_count=0,
        confidence_mean=1.0,
        confidence_min=1.0,
        confidence_max=1.0,
    )
    route = KnowledgeEdge(
        edge_id="edge-route",
        head_node_id="drug",
        tail_node_id="route",
        relation_type="HAS_ROUTE",
        support_count=2,
        document_count=2,
        confidence_mean=1.0,
        confidence_min=1.0,
        confidence_max=1.0,
        sources=("fixture",),
        layers=("silver",),
    )
    evidence = KnowledgeEvidence(
        evidence_id="evidence-route",
        edge_id=route.edge_id,
        source_record_id="relation-1",
        source_record_kind="mined_relation",
        source="fixture",
        document_id="doc-1",
        source_artifact_id="artifact-1",
        evidence_span=(0, 10),
    )
    nodes_path = tmp_path / "nodes.jsonl"
    edges_path = tmp_path / "edges.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    write_jsonl(nodes_path, (node.to_dict() for node in nodes.values()))
    write_jsonl(edges_path, (edge.to_dict() for edge in (hierarchy, route)))
    write_jsonl(evidence_path, (evidence.to_dict(),))
    return nodes_path, edges_path, evidence_path, nodes


def test_sqlite_graph_supports_search_neighbors_ancestors_and_evidence(
    tmp_path: Path,
) -> None:
    nodes_path, edges_path, evidence_path, _ = _write_graph(tmp_path)
    manifest = build_knowledge_graph_index(
        nodes_path,
        edges_path,
        evidence_path,
        cache_dir=tmp_path / "cache",
    )
    repository = SQLiteKnowledgeGraphRepository(
        manifest.index_path,
        expected_nodes_path=nodes_path,
        expected_edges_path=edges_path,
        expected_evidence_path=evidence_path,
    )

    assert manifest.node_count == 4
    assert manifest.alias_count == 5
    child = repository.get_by_code("ICD-10", "I10")
    assert child is not None
    assert child.node_id == "child"
    assert repository.search_nodes("cao huyết áp")[0].node_id == "child"
    assert repository.search_nodes("cao huyet ap")[0].node_id == "child"
    neighbor = repository.neighbors("drug", relation_types=("HAS_ROUTE",))[0]
    assert neighbor.node.node_id == "route"
    assert neighbor.edge.support_count == 2
    assert repository.ancestors("child")[0][0].node_id == "root"
    assert repository.evidence("edge-route")[0].document_id == "doc-1"
    repository.close()


def test_sqlite_graph_concurrent_reads_are_deterministic(tmp_path: Path) -> None:
    nodes_path, edges_path, evidence_path, _ = _write_graph(tmp_path)
    manifest = build_knowledge_graph_index(nodes_path, edges_path, evidence_path, cache_dir=tmp_path)
    repository = SQLiteKnowledgeGraphRepository(manifest.index_path)

    def query(_: int) -> tuple[str, ...]:
        return tuple(node.node_id for node in repository.search_nodes("cao huyết áp"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(query, range(32)))

    assert results == [("child",)] * 32
    repository.close()


def test_graph_exact_retriever_is_type_filtered_and_does_not_use_fts(tmp_path: Path) -> None:
    nodes_path, edges_path, evidence_path, _ = _write_graph(tmp_path)
    manifest = build_knowledge_graph_index(nodes_path, edges_path, evidence_path, cache_dir=tmp_path)
    repository = SQLiteKnowledgeGraphRepository(manifest.index_path)
    adapter = KnowledgeGraphExactRetrieverAdapter(repository)

    candidates = adapter.retrieve("cao huyet ap", EntityType.DISEASE, "", 5)
    assert [(item.code_system.value, item.code) for item in candidates] == [("ICD-10", "I10")]
    assert adapter.retrieve("hypertension disease", EntityType.DISEASE, "", 5) == []
    assert adapter.retrieve("cao huyết áp", EntityType.DRUG, "", 5) == []
    repository.close()


def test_graph_alias_benchmark_reports_coverage_and_rank(tmp_path: Path) -> None:
    nodes_path, edges_path, evidence_path, _ = _write_graph(tmp_path)
    manifest = build_knowledge_graph_index(nodes_path, edges_path, evidence_path, cache_dir=tmp_path)
    overlay = tmp_path / "aliases.jsonl"
    write_jsonl(
        overlay,
        (
            {
                "alias": "cao huyết áp",
                "target_concept_id": "ICD10:I10",
                "semantic_type": "DISEASE",
            },
            {
                "alias": "missing",
                "target_concept_id": "ICD10:Z99.9",
                "semantic_type": "DISEASE",
            },
        ),
    )
    repository = SQLiteKnowledgeGraphRepository(manifest.index_path)
    report = benchmark_graph_aliases(repository, (overlay,), limit=5)
    source = report["sources"][str(overlay)]
    assert source["rows"] == 2
    assert source["covered_targets"] == 1
    assert source["unknown_targets"] == 1
    assert source["top1_rate"] == 1.0
    repository.close()


def test_graph_relation_benchmark_checks_concurrent_traversal(tmp_path: Path) -> None:
    nodes_path, edges_path, evidence_path, _ = _write_graph(tmp_path)
    manifest = build_knowledge_graph_index(
        nodes_path, edges_path, evidence_path, cache_dir=tmp_path
    )
    repository = SQLiteKnowledgeGraphRepository(manifest.index_path)

    report = benchmark_graph_relations(
        repository,
        edges_path,
        relation_type="HAS_ROUTE",
        workers=4,
        repeats=3,
    )

    assert report["expected_edge_count"] == 1
    assert report["coverage_rate"] == 1.0
    assert report["deterministic"] is True
    assert report["sample_misses"] == []
    repository.close()


def test_sqlite_graph_rejects_stale_inputs_and_unknown_endpoints(tmp_path: Path) -> None:
    nodes_path, edges_path, evidence_path, nodes = _write_graph(tmp_path)
    manifest = build_knowledge_graph_index(nodes_path, edges_path, evidence_path, cache_dir=tmp_path)
    write_jsonl(nodes_path, (node.to_dict() for node in (*nodes.values(), _node("extra", "x", "OTHER"))))

    with pytest.raises(ValueError, match="fingerprint is stale"):
        SQLiteKnowledgeGraphRepository(
            manifest.index_path,
            expected_nodes_path=nodes_path,
            expected_edges_path=edges_path,
            expected_evidence_path=evidence_path,
        )

    bad_edge = KnowledgeEdge(
        edge_id="bad",
        head_node_id="extra",
        tail_node_id="missing",
        relation_type="ASSOCIATED_WITH",
        support_count=1,
        document_count=0,
        confidence_mean=1.0,
        confidence_min=1.0,
        confidence_max=1.0,
    )
    write_jsonl(edges_path, (bad_edge.to_dict(),))
    write_jsonl(evidence_path, ())
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        build_knowledge_graph_index(
            nodes_path,
            edges_path,
            evidence_path,
            output_path=tmp_path / "bad.sqlite3",
        )


@pytest.mark.integration
def test_kg_cli_builds_manifest_and_queries_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    nodes_path, edges_path, evidence_path, _ = _write_graph(tmp_path)
    index = tmp_path / "graph.sqlite3"
    manifest = tmp_path / "manifest.json"

    assert (
        main(
            [
                "kg",
                "build",
                "--nodes",
                str(nodes_path),
                "--edges",
                str(edges_path),
                "--evidence",
                str(evidence_path),
                "--output",
                str(index),
                "--manifest-output",
                str(manifest),
            ]
        )
        == 0
    )
    build_payload = json.loads(capsys.readouterr().out)
    assert build_payload["node_count"] == 4
    assert json.loads(manifest.read_text(encoding="utf-8")) == build_payload

    assert (
        main(
            [
                "kg",
                "inspect",
                "--index",
                str(index),
                "--code-system",
                "ICD-10",
                "--code",
                "I10",
            ]
        )
        == 0
    )
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["code_result"]["node_id"] == "child"

    overlay = tmp_path / "cli-aliases.jsonl"
    write_jsonl(
        overlay,
        (
            {
                "alias": "cao huyết áp",
                "target_concept_id": "ICD10:I10",
                "semantic_type": "DISEASE",
            },
        ),
    )
    report_path = tmp_path / "alias-report.json"
    assert (
        main(
            [
                "kg",
                "benchmark-aliases",
                "--index",
                str(index),
                "--alias-overlay",
                str(overlay),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    benchmark_payload = json.loads(capsys.readouterr().out)
    assert benchmark_payload["sources"][str(overlay)]["top1_rate"] == 1.0
