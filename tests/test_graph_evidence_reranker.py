"""Contracts for evidence-only knowledge-graph candidate reranking."""

from __future__ import annotations

from pathlib import Path

from medical_kg_nlp.kg.knowledge_schema import KnowledgeEdge, KnowledgeNode, KnowledgeNodeKind
from medical_kg_nlp.kg.sqlite_builder import build_knowledge_graph_index
from medical_kg_nlp.kg.sqlite_repository import SQLiteKnowledgeGraphRepository
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.graph_evidence import GraphContextConcept, GraphEvidenceReranker
from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match


def test_graph_evidence_promotes_only_an_existing_candidate(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    reranker = GraphEvidenceReranker(repository, min_support=2, max_bonus=0.05)
    candidates = [_candidate("B", 0.49), _candidate("C", 0.50)]

    reranked = reranker.rerank(
        candidates,
        context_concepts=(GraphContextConcept(CodeSystem.ICD10, "A"),),
    )

    assert [candidate.code for candidate in reranked] == ["B", "C"]
    assert {candidate.code for candidate in reranked} == {"B", "C"}
    assert "graph_reranker" in reranked[0].sources
    repository.close()


def test_graph_evidence_respects_support_gate_and_empty_context(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    candidate = _candidate("C", 0.5)
    reranker = GraphEvidenceReranker(repository, min_support=2, max_bonus=0.1)

    assert reranker.rerank([candidate]) == [candidate]
    assert reranker.rerank(
        [candidate],
        context_concepts=(GraphContextConcept(CodeSystem.ICD10, "A"),),
    ) == [candidate]
    repository.close()


def test_graph_evidence_is_deterministic_and_cached(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    reranker = GraphEvidenceReranker(repository, min_support=1, max_bonus=0.05)
    context = (GraphContextConcept(CodeSystem.ICD10, "A"),)
    candidates = [_candidate("B", 0.5), _candidate("C", 0.5)]

    first = reranker.rerank(candidates, context_concepts=context)
    second = reranker.rerank(candidates, context_concepts=context)

    assert first == second
    assert [candidate.code for candidate in first] == ["B", "C"]
    repository.close()


def _repository(tmp_path: Path) -> SQLiteKnowledgeGraphRepository:
    nodes = [
        _node("node-a", "A"),
        _node("node-b", "B"),
        _node("node-c", "C"),
    ]
    edges = [
        _edge("edge-ab", "node-a", "node-b", support=4),
        _edge("edge-ac", "node-a", "node-c", support=1),
    ]
    nodes_path = tmp_path / "nodes.jsonl"
    edges_path = tmp_path / "edges.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    write_jsonl(nodes_path, (node.to_dict() for node in nodes))
    write_jsonl(edges_path, (edge.to_dict() for edge in edges))
    write_jsonl(evidence_path, ())
    manifest = build_knowledge_graph_index(
        nodes_path,
        edges_path,
        evidence_path,
        cache_dir=tmp_path / "cache",
    )
    return SQLiteKnowledgeGraphRepository(manifest.index_path)


def _node(node_id: str, code: str) -> KnowledgeNode:
    return KnowledgeNode(
        node_id=node_id,
        kind=KnowledgeNodeKind.CONCEPT,
        label=f"Disease {code}",
        normalized_label=normalize_for_match(f"Disease {code}"),
        entity_type=EntityType.DISEASE.value,
        code_system=CodeSystem.ICD10.value,
        code=code,
    )


def _edge(
    edge_id: str,
    head: str,
    tail: str,
    *,
    support: int,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        edge_id=edge_id,
        head_node_id=head,
        tail_node_id=tail,
        relation_type="CO_OCCURS_WITH",
        support_count=support,
        document_count=support,
        confidence_mean=1.0,
        confidence_min=1.0,
        confidence_max=1.0,
    )


def _candidate(code: str, score: float) -> Candidate:
    return Candidate(
        concept_id=f"ICD10:{code}",
        code=code,
        code_system=CodeSystem.ICD10,
        canonical_name=f"Disease {code}",
        semantic_type=EntityType.DISEASE,
        score=score,
        source="bm25",
    )
