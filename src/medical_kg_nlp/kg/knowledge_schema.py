"""Persistent, provenance-bearing records for compiled medical knowledge graphs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "KnowledgeEdge",
    "KnowledgeEvidence",
    "KnowledgeNode",
    "KnowledgeNodeKind",
]


class KnowledgeNodeKind(str, Enum):
    """Stable identity class for coded concepts and uncoded normalized terms."""

    CONCEPT = "CONCEPT"
    TERM = "TERM"


@dataclass(frozen=True)
class KnowledgeNode:
    """One deduplicated graph node with terminology and corpus support."""

    node_id: str
    kind: KnowledgeNodeKind
    label: str
    normalized_label: str
    entity_type: str
    code_system: str | None = None
    code: str | None = None
    aliases: tuple[str, ...] = ()
    terminology_versions: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    occurrence_count: int = 0
    document_count: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("node_id", self.node_id),
            ("label", self.label),
            ("normalized_label", self.normalized_label),
            ("entity_type", self.entity_type),
        ):
            if not value.strip():
                raise ValueError(f"Knowledge node {name} must be non-empty")
        if self.kind == KnowledgeNodeKind.CONCEPT:
            if not self.code_system or not self.code:
                raise ValueError("Concept nodes require code_system and code")
        elif self.code_system is not None or self.code is not None:
            raise ValueError("Term nodes cannot carry a medical code")
        if self.occurrence_count < 0 or self.document_count < 0:
            raise ValueError("Knowledge node support counts must be non-negative")
        if self.document_count > self.occurrence_count:
            raise ValueError("Knowledge node document_count cannot exceed occurrence_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "label": self.label,
            "normalized_label": self.normalized_label,
            "entity_type": self.entity_type,
            "code_system": self.code_system,
            "code": self.code,
            "aliases": list(self.aliases),
            "terminology_versions": list(self.terminology_versions),
            "sources": list(self.sources),
            "occurrence_count": self.occurrence_count,
            "document_count": self.document_count,
        }


@dataclass(frozen=True)
class KnowledgeEdge:
    """One semantic edge aggregated across source records and documents."""

    edge_id: str
    head_node_id: str
    tail_node_id: str
    relation_type: str
    support_count: int
    document_count: int
    confidence_mean: float
    confidence_min: float
    confidence_max: float
    sources: tuple[str, ...] = ()
    layers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("edge_id", self.edge_id),
            ("head_node_id", self.head_node_id),
            ("tail_node_id", self.tail_node_id),
            ("relation_type", self.relation_type),
        ):
            if not value.strip():
                raise ValueError(f"Knowledge edge {name} must be non-empty")
        if self.head_node_id == self.tail_node_id:
            raise ValueError("Knowledge graph self-edges are not useful")
        if self.support_count < 1 or self.document_count < 0:
            raise ValueError("Knowledge edge support counts are invalid")
        if self.document_count > self.support_count:
            raise ValueError("Knowledge edge document_count cannot exceed support_count")
        for confidence in (self.confidence_mean, self.confidence_min, self.confidence_max):
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("Knowledge edge confidence must be in [0, 1]")
        if not self.confidence_min <= self.confidence_mean <= self.confidence_max:
            raise ValueError("Knowledge edge confidence summary is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "head_node_id": self.head_node_id,
            "tail_node_id": self.tail_node_id,
            "relation_type": self.relation_type,
            "support_count": self.support_count,
            "document_count": self.document_count,
            "confidence_mean": self.confidence_mean,
            "confidence_min": self.confidence_min,
            "confidence_max": self.confidence_max,
            "sources": list(self.sources),
            "layers": list(self.layers),
        }


@dataclass(frozen=True)
class KnowledgeEvidence:
    """Auditable source evidence retained separately from a deduplicated edge."""

    evidence_id: str
    edge_id: str
    source_record_id: str
    source_record_kind: str
    source: str
    document_id: str | None = None
    source_artifact_id: str | None = None
    evidence_span: tuple[int, int] | None = None
    head_annotation_id: str | None = None
    tail_annotation_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_id", self.evidence_id),
            ("edge_id", self.edge_id),
            ("source_record_id", self.source_record_id),
            ("source_record_kind", self.source_record_kind),
            ("source", self.source),
        ):
            if not value.strip():
                raise ValueError(f"Knowledge evidence {name} must be non-empty")
        if self.evidence_span is not None:
            start, end = self.evidence_span
            if start < 0 or end <= start:
                raise ValueError("Knowledge evidence span is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "edge_id": self.edge_id,
            "source_record_id": self.source_record_id,
            "source_record_kind": self.source_record_kind,
            "source": self.source,
            "document_id": self.document_id,
            "source_artifact_id": self.source_artifact_id,
            "evidence_span": list(self.evidence_span) if self.evidence_span else None,
            "head_annotation_id": self.head_annotation_id,
            "tail_annotation_id": self.tail_annotation_id,
        }
