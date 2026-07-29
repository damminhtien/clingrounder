"""Conflict taxonomy and global utility selection for Phase 1 proposals."""

from __future__ import annotations

from medical_kg_nlp.benchmarks.phase1.proposal_conflict_graph import (
    Phase1ConflictKind,
    Phase1ConflictNode,
    build_phase1_conflict_graph,
    select_maximum_utility_nodes,
)


def test_conflict_graph_groups_boundary_type_and_containment_alternatives() -> None:
    graph = build_phase1_conflict_graph(
        (
            _node("disease", (0, 8), "CHẨN_ĐOÁN"),
            _node("symptom", (0, 8), "TRIỆU_CHỨNG"),
            _node("short", (0, 4), "TRIỆU_CHỨNG"),
            _node("crossing", (6, 12), "TRIỆU_CHỨNG"),
        )
    )

    assert len(graph.components) == 1
    assert {edge.kind for edge in graph.edges} == {
        Phase1ConflictKind.TYPE_CONFLICT,
        Phase1ConflictKind.CONTAINMENT,
        Phase1ConflictKind.BOUNDARY_OVERLAP,
    }


def test_conflict_graph_identifies_drug_lab_result_ambiguity() -> None:
    graph = build_phase1_conflict_graph(
        (
            _node("drug", (0, 18), "THUỐC"),
            _node("result", (8, 13), "KẾT_QUẢ_XÉT_NGHIỆM"),
        )
    )

    assert graph.edges[0].kind is Phase1ConflictKind.DOSAGE_LAB_AMBIGUITY


def test_weighted_scheduler_beats_greedy_broad_span() -> None:
    graph = build_phase1_conflict_graph(
        (
            _node("broad", (0, 17), "TRIỆU_CHỨNG", probability=0.9),
            _node("left", (0, 8), "TRIỆU_CHỨNG", probability=0.8),
            _node("right", (9, 17), "TRIỆU_CHỨNG", probability=0.8),
        )
    )

    selected = select_maximum_utility_nodes(graph)

    assert [node.node_id for node in selected] == ["left", "right"]


def test_weighted_scheduler_is_deterministic_on_equal_utility() -> None:
    graph = build_phase1_conflict_graph(
        (
            _node("later-id", (0, 4), "CHẨN_ĐOÁN"),
            _node("earlier-id", (0, 4), "TRIỆU_CHỨNG"),
        )
    )

    first = select_maximum_utility_nodes(graph)
    second = select_maximum_utility_nodes(graph)

    assert first == second
    assert len(first) == 1


def test_weighted_scheduler_respects_calibrated_threshold_below_half() -> None:
    graph = build_phase1_conflict_graph(
        (
            _node(
                "recall-oriented",
                (0, 8),
                "TRIỆU_CHỨNG",
                probability=0.4,
                decision_threshold=0.3,
            ),
        )
    )

    selected = select_maximum_utility_nodes(graph)

    assert [node.node_id for node in selected] == ["recall-oriented"]


def _node(
    node_id: str,
    span: tuple[int, int],
    entity_type: str,
    *,
    probability: float = 0.8,
    decision_threshold: float = 0.5,
) -> Phase1ConflictNode:
    return Phase1ConflictNode(
        node_id=node_id,
        document_id="1",
        span=span,
        entity_type=entity_type,
        probability=probability,
        source_count=1,
        decision_threshold=decision_threshold,
    )
