from __future__ import annotations

import pytest

from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.training.listwise_linking import (
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
    with pytest.raises(ValueError, match="fix recall before reranking"):
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
