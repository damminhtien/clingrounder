"""Regression tests for split-safe graph reranker benchmarking."""

from __future__ import annotations

from pathlib import Path

import pytest

from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.kg.knowledge_schema import KnowledgeEdge, KnowledgeNode, KnowledgeNodeKind
from medical_kg_nlp.kg.reranker_benchmark import benchmark_graph_candidate_reranking
from medical_kg_nlp.kg.sqlite_builder import build_knowledge_graph_index
from medical_kg_nlp.kg.sqlite_repository import SQLiteKnowledgeGraphRepository
from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match


def test_graph_reranker_benchmark_cli_is_discoverable() -> None:
    args = build_parser().parse_args(
        [
            "kg",
            "benchmark-reranker",
            "--index",
            "graph.sqlite3",
            "--nodes",
            "nodes.jsonl",
            "--edges",
            "edges.jsonl",
            "--evidence",
            "evidence.jsonl",
            "--terminology-index",
            "terminology.sqlite3",
            "--terminology-source",
            "concepts.jsonl",
            "--documents",
            "documents.jsonl",
            "--annotations",
            "annotations.jsonl",
            "--output",
            "report.json",
        ]
    )

    assert args.handler == "kg_benchmark_reranker"
    assert args.max_bonus_grid == [0.0, 0.01, 0.02, 0.04, 0.08]
    assert args.context_mode == "oracle"


def test_graph_reranker_calibrates_on_dev_and_improves_heldout_test(tmp_path: Path) -> None:
    graph, evidence = _graph(tmp_path)
    documents = tmp_path / "documents.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl(
        documents,
        (
            _document("train", "train", "source"),
            _document("dev", "dev", "context ambiguous."),
            _document("test", "test", "context ambiguous."),
        ),
    )
    write_jsonl(
        annotations,
        (
            *_annotations("dev", "context ambiguous."),
            *_annotations("test", "context ambiguous."),
        ),
    )

    report = benchmark_graph_candidate_reranking(
        graph,
        _TerminologyFixture(),
        documents_path=documents,
        annotations_path=annotations,
        graph_evidence_path=evidence,
        max_bonus_grid=(0.0, 0.04),
        min_support=2,
    )

    assert report["feature"]["selected_max_bonus"] == 0.04
    assert report["evaluation"]["delta"]["accuracy_at_1"] == 0.5
    assert report["evaluation"]["rank_changes"]["worsened"] == 0
    assert report["leakage_contract"]["status"] == "passed"
    graph.close()


def test_graph_reranker_uses_predicted_exact_unique_context(tmp_path: Path) -> None:
    graph, evidence = _graph(tmp_path)
    documents = tmp_path / "documents.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl(
        documents,
        (
            _document("train", "train", "source"),
            _document("dev", "dev", "context ambiguous."),
            _document("test", "test", "context ambiguous."),
        ),
    )
    write_jsonl(
        annotations,
        (
            *_annotations("dev", "context ambiguous."),
            *_annotations("test", "context ambiguous."),
        ),
    )

    report = benchmark_graph_candidate_reranking(
        graph,
        _TerminologyFixture(),
        documents_path=documents,
        annotations_path=annotations,
        graph_evidence_path=evidence,
        context_mode="predicted_exact_unique",
        max_bonus_grid=(0.0, 0.04),
        min_support=2,
    )

    assert report["semantic_contract"] == (
        "gold_mentions_with_predicted_exact_unique_context_links"
    )
    assert report["evaluation"]["context_anchors"] == {
        "mode": "predicted_exact_unique",
        "emitted": 1,
        "correct": 1,
        "precision": 1.0,
    }
    assert report["evaluation"]["delta"]["accuracy_at_1"] == 0.5
    graph.close()


def test_graph_reranker_rejects_evaluation_evidence(tmp_path: Path) -> None:
    graph, evidence = _graph(tmp_path, evidence_document="codiesp:dev")
    documents = tmp_path / "documents.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl(
        documents,
        (
            _document("train", "train", "source"),
            _document("dev", "dev", "context ambiguous."),
            _document("test", "test", "context ambiguous."),
        ),
    )
    write_jsonl(annotations, ())

    with pytest.raises(ValueError, match="forbidden split dev"):
        benchmark_graph_candidate_reranking(
            graph,
            _TerminologyFixture(),
            documents_path=documents,
            annotations_path=annotations,
            graph_evidence_path=evidence,
        )
    graph.close()


def _entry(code: str, name: str) -> ConceptEntry:
    return ConceptEntry(
        concept_id=f"ICD10:{code}",
        code=code,
        code_system=CodeSystem.ICD10,
        canonical_name=name,
        semantic_type=EntityType.DISEASE,
    )


class _TerminologyFixture:
    entries = {
        "A": _entry("A", "context"),
        "B": _entry("B", "expected"),
        "C": _entry("C", "distractor"),
    }

    def get_by_concept_id(self, concept_id: str) -> ConceptEntry | None:
        return next(
            (entry for entry in self.entries.values() if entry.concept_id == concept_id),
            None,
        )

    def get_by_code(self, code_system: CodeSystem, code: str) -> ConceptEntry | None:
        return self.entries.get(code) if code_system == CodeSystem.ICD10 else None

    def exact_lookup(self, mention: str, **kwargs: object) -> list[ConceptEntry]:
        del kwargs
        return [self.entries["A"]] if mention == "context" else []

    def toneless_lookup(self, mention: str, **kwargs: object) -> list[ConceptEntry]:
        del mention, kwargs
        return []

    def search(self, mention: str, **kwargs: object) -> list[ConceptEntry]:
        del kwargs
        if mention == "ambiguous":
            return [self.entries["C"], self.entries["B"]]
        return [self.entries["A"]] if mention == "context" else []
def _graph(
    tmp_path: Path,
    *,
    evidence_document: str = "codiesp:train",
) -> tuple[SQLiteKnowledgeGraphRepository, Path]:
    nodes = [_node("node-a", "A"), _node("node-b", "B"), _node("node-c", "C")]
    edge = KnowledgeEdge(
        edge_id="edge-ab",
        head_node_id="node-a",
        tail_node_id="node-b",
        relation_type="CO_OCCURS_WITH",
        support_count=4,
        document_count=4,
        confidence_mean=1.0,
        confidence_min=1.0,
        confidence_max=1.0,
    )
    nodes_path = tmp_path / "nodes.jsonl"
    edges_path = tmp_path / "edges.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    write_jsonl(nodes_path, (node.to_dict() for node in nodes))
    write_jsonl(edges_path, (edge.to_dict(),))
    write_jsonl(
        evidence_path,
        (
            {
                "document_id": evidence_document,
                "edge_id": "edge-ab",
                "evidence_id": "evidence-ab",
                "evidence_span": [0, 6],
                "head_annotation_id": "head",
                "source": "fixture",
                "source_artifact_id": "artifact",
                "source_record_id": "relation",
                "source_record_kind": "mined_relation",
                "tail_annotation_id": "tail",
            },
        ),
    )
    manifest = build_knowledge_graph_index(
        nodes_path,
        edges_path,
        evidence_path,
        cache_dir=tmp_path / "cache",
    )
    return SQLiteKnowledgeGraphRepository(manifest.index_path), evidence_path


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


def _document(suffix: str, split: str, text: str) -> dict[str, object]:
    return {
        "document_id": f"codiesp:{suffix}",
        "metadata": {"corpus_split": split},
        "text": text,
    }


def _annotations(suffix: str, text: str) -> tuple[dict[str, object], ...]:
    return (
        _annotation(suffix, "context", 0, 7, "A"),
        _annotation(suffix, "ambiguous", 8, 17, "B"),
    )


def _annotation(
    suffix: str,
    text: str,
    start: int,
    end: int,
    code: str,
) -> dict[str, object]:
    return {
        "annotation_id": f"{suffix}:{code}",
        "concepts": [{"code": code, "code_system": "ICD-10"}],
        "document_id": f"codiesp:{suffix}",
        "entity_type": "DISEASE",
        "metadata": {"discontinuous": "false", "source_text_match": "true"},
        "source_label": "DIAGNOSTICO",
        "span": [start, end],
        "text": text,
    }
