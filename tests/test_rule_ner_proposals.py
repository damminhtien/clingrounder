from __future__ import annotations

import pytest

from medical_kg_nlp.ner.proposal import EntityProposal
from medical_kg_nlp.ner.span_resolver import EvidenceWeightedSpanResolver
from medical_kg_nlp.schema.types import EntityType


def test_entity_proposal_requires_sorted_unique_candidate_types() -> None:
    with pytest.raises(ValueError, match="candidate_types"):
        EntityProposal(
            span=(0, 4),
            candidate_types=(EntityType.SYMPTOM, EntityType.DISEASE),
            source="dictionary",
            score=0.8,
        )


def test_span_resolver_allows_late_stronger_source_to_win_overlap() -> None:
    proposals = (
        _proposal((0, 12), EntityType.DISEASE, "dictionary_exact", 0.76),
        _proposal((0, 8), EntityType.SYMPTOM, "structured_symptom", 0.95),
    )

    result = EvidenceWeightedSpanResolver().resolve(proposals)

    assert [(item.span, item.entity_type, item.source) for item in result.selected] == [
        ((0, 8), EntityType.SYMPTOM, "structured_symptom")
    ]
    assert {
        (decision.source, decision.accepted, decision.reason)
        for decision in result.decisions
    } == {
        ("dictionary_exact", False, "rejected_overlap"),
        ("structured_symptom", True, "selected_global_utility"),
    }


def test_span_resolver_prefers_two_independent_mentions_to_one_broad_alias() -> None:
    proposals = (
        _proposal((0, 17), EntityType.SYMPTOM, "dictionary_exact", 0.8),
        _proposal((0, 8), EntityType.SYMPTOM, "dictionary_exact", 0.8),
        _proposal((9, 17), EntityType.SYMPTOM, "dictionary_exact", 0.8),
    )

    result = EvidenceWeightedSpanResolver().resolve(proposals)

    assert [item.span for item in result.selected] == [(0, 8), (9, 17)]


def test_span_resolver_merges_exact_source_agreement() -> None:
    proposals = (
        _proposal((4, 12), EntityType.SYMPTOM, "dictionary_exact", 0.82),
        _proposal((4, 12), EntityType.SYMPTOM, "lab_grammar", 0.81),
    )

    result = EvidenceWeightedSpanResolver().resolve(proposals)

    assert len(result.selected) == 1
    accepted = [decision for decision in result.decisions if decision.accepted]
    assert len(accepted) == 2
    assert {decision.reason for decision in accepted} == {"selected_exact_consensus"}


def test_span_resolver_retains_ambiguous_proposal_as_rejected_evidence() -> None:
    ambiguous = EntityProposal(
        span=(0, 4),
        candidate_types=(EntityType.DISEASE, EntityType.SYMPTOM),
        source="dictionary_exact",
        score=0.8,
    )

    result = EvidenceWeightedSpanResolver().resolve((ambiguous,))

    assert result.selected == ()
    assert result.decisions[0].reason == "unresolved_type"


def _proposal(
    span: tuple[int, int],
    entity_type: EntityType,
    source: str,
    score: float,
) -> EntityProposal:
    return EntityProposal(
        span=span,
        candidate_types=(entity_type,),
        source=source,
        score=score,
    )
