from __future__ import annotations
from medical_kg_nlp.kg.graph_schema import KGEdge, KGNode


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.nodes: dict[str, KGNode] = {}
        self.edges: list[KGEdge] = []

    def add_node(self, node: KGNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: KGEdge) -> None:
        self.edges.append(edge)

