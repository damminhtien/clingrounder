"""Public knowledge-graph records plus lazy ontology and validator access."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from medical_kg_nlp.kg.knowledge_schema import (
    KnowledgeEdge,
    KnowledgeEvidence,
    KnowledgeNeighbor,
    KnowledgeNode,
    KnowledgeNodeKind,
)
from medical_kg_nlp.kg.ports import KnowledgeGraphRepositoryPort
from medical_kg_nlp.kg.sqlite_builder import (
    KNOWLEDGE_GRAPH_SCHEMA_VERSION,
    KnowledgeGraphIndexManifest,
    build_knowledge_graph_index,
)
from medical_kg_nlp.kg.sqlite_repository import (
    SQLiteKnowledgeGraphRepository,
)

if TYPE_CHECKING:
    from medical_kg_nlp.kg.ontology_reasoner import OntologyReasoner
    from medical_kg_nlp.kg.validator import KGValidator

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
]


def __getattr__(name: str) -> Any:
    if name == "KGValidator":
        from medical_kg_nlp.kg.validator import KGValidator

        return KGValidator
    if name == "OntologyReasoner":
        from medical_kg_nlp.kg.ontology_reasoner import OntologyReasoner

        return OntologyReasoner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
