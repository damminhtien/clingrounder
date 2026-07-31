"""Conflict components and global interval selection for calibrated proposals."""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "Phase1ConflictComponent",
    "Phase1ConflictEdge",
    "Phase1ConflictGraph",
    "Phase1ConflictKind",
    "Phase1ConflictNode",
    "build_phase1_conflict_graph",
    "select_maximum_utility_nodes",
]

_DRUG = "THUỐC"
_LAB_RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"


class Phase1ConflictKind(StrEnum):
    """Auditable reasons why two proposals cannot both enter the output."""

    BOUNDARY_OVERLAP = "boundary_overlap"
    CONTAINMENT = "containment"
    DOSAGE_LAB_AMBIGUITY = "dosage_lab_ambiguity"
    DUPLICATE_EXACT = "duplicate_exact"
    TYPE_CONFLICT = "type_conflict"


@dataclass(frozen=True, slots=True)
class Phase1ConflictNode:
    """One exact-span/type alternative with a calibrated correctness probability."""

    node_id: str
    document_id: str
    span: tuple[int, int]
    entity_type: str
    probability: float
    source_count: int
    decision_threshold: float = 0.5

    def __post_init__(self) -> None:
        start, end = self.span
        if not self.node_id.strip() or not self.document_id.strip():
            raise ValueError("Conflict nodes require node_id and document_id")
        if start < 0 or end <= start:
            raise ValueError("Conflict node span must be non-empty")
        if not self.entity_type.strip():
            raise ValueError("Conflict node entity_type must be non-empty")
        if (
            not math.isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError("Conflict node probability must be within [0, 1]")
        if (
            not math.isfinite(self.decision_threshold)
            or not 0.0 <= self.decision_threshold <= 1.0
        ):
            raise ValueError("Conflict node decision threshold must be within [0, 1]")
        if self.source_count <= 0:
            raise ValueError("Conflict node source_count must be positive")

    @property
    def utility(self) -> float:
        """Return calibrated log-odds margin used by weighted interval scheduling."""

        # MODEL: the operating point may be below 0.5 for a recall-oriented entity type.
        # Subtracting its logit prevents the conflict resolver from silently reintroducing
        # a fixed 0.5 threshold after the calibrated gate has accepted a proposal.
        return _logit(self.probability) - _logit(self.decision_threshold)


@dataclass(frozen=True, slots=True)
class Phase1ConflictEdge:
    """One undirected incompatibility edge."""

    left_id: str
    right_id: str
    kind: Phase1ConflictKind

    def __post_init__(self) -> None:
        if not self.left_id.strip() or not self.right_id.strip():
            raise ValueError("Conflict edges require node IDs")
        if self.left_id >= self.right_id:
            raise ValueError("Conflict edge IDs must be unique and sorted")


@dataclass(frozen=True, slots=True)
class Phase1ConflictComponent:
    """Connected alternatives that can be adjudicated without the full document."""

    component_id: str
    node_ids: tuple[str, ...]
    edges: tuple[Phase1ConflictEdge, ...]


@dataclass(frozen=True, slots=True)
class Phase1ConflictGraph:
    """Deterministic conflict graph plus connected-component lookup."""

    nodes: tuple[Phase1ConflictNode, ...]
    edges: tuple[Phase1ConflictEdge, ...]
    components: tuple[Phase1ConflictComponent, ...]

    @property
    def component_by_node_id(self) -> dict[str, Phase1ConflictComponent]:
        """Return a node-to-component view for decision traces."""

        return {
            node_id: component
            for component in self.components
            for node_id in component.node_ids
        }


def build_phase1_conflict_graph(
    nodes: tuple[Phase1ConflictNode, ...],
) -> Phase1ConflictGraph:
    """Build overlap edges with a sweep per document and group connected alternatives."""

    if len({node.node_id for node in nodes}) != len(nodes):
        raise ValueError("Conflict graph node IDs must be unique")
    ordered = tuple(sorted(nodes, key=_node_start_order))
    edges: list[Phase1ConflictEdge] = []
    active_by_document: dict[str, list[Phase1ConflictNode]] = {}
    for node in ordered:
        active = [
            other
            for other in active_by_document.get(node.document_id, [])
            if other.span[1] > node.span[0]
        ]
        for other in active:
            if _overlap(node.span, other.span):
                left_id, right_id = sorted((node.node_id, other.node_id))
                edges.append(
                    Phase1ConflictEdge(
                        left_id=left_id,
                        right_id=right_id,
                        kind=_conflict_kind(other, node),
                    )
                )
        active.append(node)
        active_by_document[node.document_id] = active
    sorted_edges = tuple(sorted(edges, key=_edge_order))
    components = _connected_components(ordered, sorted_edges)
    return Phase1ConflictGraph(
        nodes=ordered,
        edges=sorted_edges,
        components=components,
    )


def select_maximum_utility_nodes(
    graph: Phase1ConflictGraph,
) -> tuple[Phase1ConflictNode, ...]:
    """Select the deterministic maximum-log-odds non-overlapping proposal set."""

    selected: list[Phase1ConflictNode] = []
    nodes_by_document: dict[str, list[Phase1ConflictNode]] = {}
    for node in graph.nodes:
        nodes_by_document.setdefault(node.document_id, []).append(node)
    for document_id in sorted(nodes_by_document, key=_document_sort_key):
        candidates = tuple(
            sorted(nodes_by_document[document_id], key=_node_end_order)
        )
        ends = [candidate.span[1] for candidate in candidates]
        best: list[_Selection] = [_Selection()]
        for index, candidate in enumerate(candidates):
            predecessor = bisect_right(ends, candidate.span[0], hi=index)
            included = best[predecessor].append(candidate)
            excluded = best[index]
            best.append(_better_selection(included, excluded))
        selected.extend(best[-1].nodes)
    return tuple(sorted(selected, key=_node_start_order))


@dataclass(frozen=True, slots=True)
class _Selection:
    nodes: tuple[Phase1ConflictNode, ...] = ()
    utility: float = 0.0

    def append(self, node: Phase1ConflictNode) -> "_Selection":
        return _Selection(
            nodes=(*self.nodes, node),
            utility=self.utility + node.utility,
        )


def _connected_components(
    nodes: tuple[Phase1ConflictNode, ...],
    edges: tuple[Phase1ConflictEdge, ...],
) -> tuple[Phase1ConflictComponent, ...]:
    adjacency: dict[str, set[str]] = {node.node_id: set() for node in nodes}
    edge_by_node: dict[str, list[Phase1ConflictEdge]] = {
        node.node_id: [] for node in nodes
    }
    for edge in edges:
        adjacency[edge.left_id].add(edge.right_id)
        adjacency[edge.right_id].add(edge.left_id)
        edge_by_node[edge.left_id].append(edge)
        edge_by_node[edge.right_id].append(edge)
    components: list[Phase1ConflictComponent] = []
    visited: set[str] = set()
    for node in nodes:
        if node.node_id in visited:
            continue
        stack = [node.node_id]
        node_ids: set[str] = set()
        component_edges: set[Phase1ConflictEdge] = set()
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node_ids.add(node_id)
            component_edges.update(edge_by_node[node_id])
            stack.extend(sorted(adjacency[node_id] - visited, reverse=True))
        ordered_ids = tuple(sorted(node_ids))
        component_id = hashlib.sha256(
            "\n".join(ordered_ids).encode("utf-8")
        ).hexdigest()[:16]
        components.append(
            Phase1ConflictComponent(
                component_id=component_id,
                node_ids=ordered_ids,
                edges=tuple(sorted(component_edges, key=_edge_order)),
            )
        )
    return tuple(
        sorted(
            components,
            key=lambda component: component.node_ids,
        )
    )


def _conflict_kind(
    left: Phase1ConflictNode,
    right: Phase1ConflictNode,
) -> Phase1ConflictKind:
    if {left.entity_type, right.entity_type} == {_DRUG, _LAB_RESULT}:
        return Phase1ConflictKind.DOSAGE_LAB_AMBIGUITY
    if left.span == right.span:
        return (
            Phase1ConflictKind.DUPLICATE_EXACT
            if left.entity_type == right.entity_type
            else Phase1ConflictKind.TYPE_CONFLICT
        )
    if _contains(left.span, right.span) or _contains(right.span, left.span):
        return Phase1ConflictKind.CONTAINMENT
    return Phase1ConflictKind.BOUNDARY_OVERLAP


def _better_selection(left: _Selection, right: _Selection) -> _Selection:
    # MODEL: source agreement and span length are already learned features. Reusing them here
    # would silently turn calibrated probability back into a hand-tuned utility.
    left_utility = round(left.utility, 12)
    right_utility = round(right.utility, 12)
    if left_utility != right_utility:
        return left if left_utility > right_utility else right
    return left if _selection_signature(left) < _selection_signature(right) else right


def _selection_signature(
    selection: _Selection,
) -> tuple[tuple[str, int, int, str, str], ...]:
    return tuple(_node_start_order(node) for node in selection.nodes)


def _node_start_order(
    node: Phase1ConflictNode,
) -> tuple[str, int, int, str, str]:
    return (
        node.document_id,
        node.span[0],
        node.span[1],
        node.entity_type,
        node.node_id,
    )


def _node_end_order(node: Phase1ConflictNode) -> tuple[int, int, str, str]:
    return node.span[1], node.span[0], node.entity_type, node.node_id


def _edge_order(edge: Phase1ConflictEdge) -> tuple[str, str, str]:
    return edge.left_id, edge.right_id, edge.kind.value


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _contains(container: tuple[int, int], inner: tuple[int, int]) -> bool:
    return (
        container != inner
        and container[0] <= inner[0]
        and inner[1] <= container[1]
    )


def _logit(value: float) -> float:
    bounded = min(1.0 - 1e-9, max(1e-9, value))
    return math.log(bounded / (1.0 - bounded))


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
