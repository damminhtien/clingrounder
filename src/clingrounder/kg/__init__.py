"""Public knowledge-graph records plus lazy ontology and validator access."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from clingrounder.kg.knowledge_schema import (
    KnowledgeEdge,
    KnowledgeEvidence,
    KnowledgeNeighbor,
    KnowledgeNode,
    KnowledgeNodeKind,
)
from clingrounder.kg.benchmark import (
    benchmark_graph_aliases,
    benchmark_graph_relations,
)
from clingrounder.kg.ports import KnowledgeGraphRepositoryPort
from clingrounder.kg.sqlite_builder import (
    KNOWLEDGE_GRAPH_SCHEMA_VERSION,
    KnowledgeGraphIndexManifest,
    build_knowledge_graph_index,
)
from clingrounder.kg.sqlite_repository import (
    SQLiteKnowledgeGraphRepository,
)

if TYPE_CHECKING:
    from clingrounder.kg.ontology_reasoner import OntologyReasoner
    from clingrounder.kg.validator import KGValidator

__all__ = [
    "KGValidator",
    "KNOWLEDGE_GRAPH_SCHEMA_VERSION",
    "KnowledgeEdge",
    "KnowledgeEvidence",
    "KnowledgeGraphIndexManifest",
    "KnowledgeGraphRepositoryPort",
    "KnowledgeNeighbor",
    "KnowledgeNode",
    "KnowledgeNodeKind",
    "OntologyReasoner",
    "SQLiteKnowledgeGraphRepository",
    "build_knowledge_graph_index",
    "benchmark_graph_aliases",
    "benchmark_graph_relations",
]


def __getattr__(name: str) -> Any:
    if name == "KGValidator":
        from clingrounder.kg.validator import KGValidator

        return KGValidator
    if name == "OntologyReasoner":
        from clingrounder.kg.ontology_reasoner import OntologyReasoner

        return OntologyReasoner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
