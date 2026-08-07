"""Storage-neutral query contract for medical knowledge graphs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from clingrounder.kg.knowledge_schema import (
    KnowledgeEvidence,
    KnowledgeNeighbor,
    KnowledgeNode,
)

__all__ = ["KnowledgeGraphRepositoryPort"]


class KnowledgeGraphRepositoryPort(Protocol):
    """Expose graph evidence without coupling consumers to a storage backend."""

    def get_node(self, node_id: str) -> KnowledgeNode | None: ...

    def get_by_code(self, code_system: str, code: str) -> KnowledgeNode | None: ...

    def search_nodes(
        self,
        query: str,
        *,
        entity_type: str | None = None,
        code_system: str | None = None,
        limit: int = 20,
        exact_only: bool = False,
    ) -> list[KnowledgeNode]: ...

    def neighbors(
        self,
        node_id: str,
        *,
        direction: str = "outgoing",
        relation_types: Sequence[str] = (),
        min_support: int = 1,
        limit: int = 100,
    ) -> list[KnowledgeNeighbor]: ...

    def ancestors(
        self,
        node_id: str,
        *,
        max_depth: int = 20,
    ) -> list[tuple[KnowledgeNode, int]]: ...

    def evidence(self, edge_id: str, *, limit: int = 100) -> list[KnowledgeEvidence]: ...

    def close(self) -> None: ...
