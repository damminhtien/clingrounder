from __future__ import annotations

import pytest

from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.listwise import (
    ListwiseLinkingQuery,
    ListwiseOrderRanking,
    aggregate_listwise_rankings,
    build_listwise_candidate_orders,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.training.listwise_linking import (
    CandidateRecallError,
    build_pairwise_linking_examples,
    build_listwise_linking_record,
    evaluate_listwise_scores,
    render_listwise_input,
    shuffle_listwise_candidates,
)


def test_listwise_record_keeps_multiple_valid_codes() -> None:
    record = build_listwise_linking_record(
        query_id="q1",
        mention="thuốc mẫu",
        context="Bệnh nhân dùng thuốc mẫu mỗi ngày.",
        entity_type=EntityType.DRUG,
        candidates=[
            _candidate("A", "1", 0.9),
            _candidate("B", "2", 0.8),
            _candidate("C", "3", 0.7),
        ],
        positive_codes=("1", "2"),
    )

    assert record.positive_indices == (0, 1)
    rendered = render_listwise_input(record)
    assert "[MENTION] thuốc mẫu" in rendered
    assert "[0] RxNorm|1|name A|alias A" in rendered


def test_listwise_shuffle_is_deterministic_and_remaps_labels() -> None:
    record = build_listwise_linking_record(
        query_id="stable",
        mention="thuốc mẫu",
        context="",
        entity_type=EntityType.DRUG,
        candidates=[
            _candidate("A", "1", 0.9),
            _candidate("B", "2", 0.8),
            _candidate("C", "3", 0.7),
        ],
        positive_codes=("2",),
    )

    first = shuffle_listwise_candidates(record, seed=13)
    second = shuffle_listwise_candidates(record, seed=13)

    assert first == second
    assert first.candidates[first.positive_indices[0]].code == "2"


def test_listwise_builder_rejects_missing_retrieval_positive() -> None:
    with pytest.raises(CandidateRecallError, match="fix recall before reranking"):
        build_listwise_linking_record(
            query_id="q1",
            mention="thuốc mẫu",
            context="",
            entity_type=EntityType.DRUG,
            candidates=[
                _candidate("A", "1", 0.9),
                _candidate("B", "2", 0.8),
            ],
            positive_codes=("missing",),
        )


def test_listwise_metrics_rank_any_valid_positive() -> None:
    record = build_listwise_linking_record(
        query_id="q1",
        mention="thuốc mẫu",
        context="",
        entity_type=EntityType.DRUG,
        candidates=[
            _candidate("A", "1", 0.9),
            _candidate("B", "2", 0.8),
            _candidate("C", "3", 0.7),
        ],
        positive_codes=("2", "3"),
    )

    report = evaluate_listwise_scores([record], {"q1": [0.1, 0.3, 0.9]})

    assert report["hit_at"]["1"] == 1.0
    assert report["mrr"] == 1.0
    assert report["top1_accuracy"] == 1.0
    assert report["jaccard_after_emission"] == 0.5


def test_pairwise_baseline_expands_one_label_per_candidate() -> None:
    record = build_listwise_linking_record(
        query_id="pairwise",
        mention="metformin",
        context="Bệnh nhân dùng metformin.",
        entity_type=EntityType.DRUG,
        candidates=[
            _candidate("A", "1", 0.9),
            _candidate("B", "2", 0.8),
            _candidate("C", "3", 0.7),
        ],
        positive_codes=("2",),
    )

    examples = build_pairwise_linking_examples((record,))

    assert [example.label for example in examples] == [0, 1, 0]
    assert all("[MENTION] metformin" in example.query_text for example in examples)
    assert "RxNorm:2" in examples[1].candidate_text


def test_three_order_aggregation_uses_global_candidate_identity() -> None:
    record = build_listwise_linking_record(
        query_id="orders",
        mention="thuốc mẫu",
        context="",
        entity_type=EntityType.DRUG,
        candidates=[
            _candidate("A", "1", 0.9),
            _candidate("B", "2", 0.8),
            _candidate("C", "3", 0.7),
        ],
        positive_codes=("2",),
    )
    query = ListwiseLinkingQuery(
        query_id=record.query_id,
        mention=record.mention,
        context=record.context,
        entity_type=record.entity_type,
        structured_mention=record.structured_mention,
        candidates=record.candidates,
    )

    orders = build_listwise_candidate_orders(query, seed=7)
    decision = aggregate_listwise_rankings(
        query,
        (
            ListwiseOrderRanking(orders[0].order_id, (1, 0, 2), False),
            ListwiseOrderRanking(orders[1].order_id, (1, 2, 0), False),
            ListwiseOrderRanking(orders[2].order_id, (1, 0, 2), False),
        ),
    )

    assert [order.order_id for order in orders] == ["retrieval", "reverse", "shuffled"]
    assert decision.ranked_candidate_indices[0] == 1
    assert decision.order_consistency == 1.0
    assert decision.abstain is False


def test_listwise_metrics_report_order_consistency_and_useful_abstention() -> None:
    record = build_listwise_linking_record(
        query_id="abstain",
        mention="thuốc mẫu",
        context="",
        entity_type=EntityType.DRUG,
        candidates=[_candidate("A", "1", 0.9), _candidate("B", "2", 0.8)],
        positive_codes=("2",),
    )

    report = evaluate_listwise_scores(
        [record],
        {"abstain": [0.9, 0.1]},
        abstained_query_ids=("abstain",),
        order_consistency={"abstain": 2 / 3},
    )

    assert report["abstention_precision"] == 1.0
    assert report["jaccard_after_emission"] == 0.0
    assert report["order_consistency"] == pytest.approx(2 / 3)


def _candidate(
    concept_id: str,
    code: str,
    score: float,
) -> Candidate:
    return Candidate(
        concept_id=concept_id,
        code=code,
        code_system=CodeSystem.RXNORM,
        canonical_name=f"name {concept_id}",
        semantic_type=EntityType.DRUG,
        score=score,
        source="exact",
        matched_alias=f"alias {concept_id}",
    )
