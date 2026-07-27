"""Global non-overlap resolution for rule-based entity proposals."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Protocol

from medical_kg_nlp.ner.proposal import EntityProposal, ProposalDecision
from medical_kg_nlp.schema.types import EntityType

__all__ = [
    "EvidenceWeightedSpanResolver",
    "ProposalScoringPort",
    "SpanResolutionResult",
]


class ProposalScoringPort(Protocol):
    """Score typed proposals for global interval selection."""

    def score(self, proposal: EntityProposal, *, source_count: int) -> float:
        """Return an additive utility; larger values are preferred."""


@dataclass(frozen=True, slots=True)
class _DefaultProposalScoring:
    """Conservative scoring until source/type probabilities are learned from reviewed data."""

    agreement_bonus: float = 0.25
    atomic_product_bonus: float = 1.5

    def score(self, proposal: EntityProposal, *, source_count: int) -> float:
        probability = min(1.0 - 1e-6, max(1e-6, proposal.score))
        utility = math.log(probability / (1.0 - probability))
        utility += max(0, source_count - 1) * self.agreement_bonus
        if proposal.feature("atomic_product") == "true":
            utility += self.atomic_product_bonus
        return utility


@dataclass(frozen=True, slots=True)
class SpanResolutionResult:
    """Selected typed proposals plus one decision for every input proposal."""

    selected: tuple[EntityProposal, ...]
    decisions: tuple[ProposalDecision, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    representative: EntityProposal
    source_proposals: tuple[EntityProposal, ...]
    utility: float


@dataclass(frozen=True, slots=True)
class _Selection:
    candidates: tuple[_Candidate, ...] = ()
    utility: float = 0.0
    source_agreement: int = 0
    covered_characters: int = 0

    def append(self, candidate: _Candidate) -> "_Selection":
        start, end = candidate.representative.span
        return _Selection(
            candidates=(*self.candidates, candidate),
            utility=self.utility + candidate.utility,
            source_agreement=self.source_agreement + len(candidate.source_proposals),
            covered_characters=self.covered_characters + end - start,
        )


@dataclass(frozen=True, slots=True)
class EvidenceWeightedSpanResolver:
    """Select a deterministic maximum-utility set after every extractor has proposed.

    INVARIANT: extractors never suppress overlaps from another source. This resolver is the only
    component allowed to reject a typed proposal because of a span conflict.

    SCALING: weighted interval scheduling is ``O(n log n)`` and avoids order-dependent pairwise
    suppression on long notes.
    """

    scoring: ProposalScoringPort = _DefaultProposalScoring()

    def resolve(self, proposals: tuple[EntityProposal, ...]) -> SpanResolutionResult:
        ambiguous = tuple(proposal for proposal in proposals if proposal.entity_type is None)
        typed = tuple(proposal for proposal in proposals if proposal.entity_type is not None)
        candidates = self._merge_exact(typed)
        selected_candidates = self._maximum_utility_set(candidates)
        selected_keys: set[tuple[tuple[int, int], EntityType]] = {
            (
                candidate.representative.span,
                _resolved_type(candidate.representative),
            )
            for candidate in selected_candidates
        }
        selected = tuple(
            sorted(
                (candidate.representative for candidate in selected_candidates),
                key=_proposal_start_order,
            )
        )
        decisions = [
            *(
                ProposalDecision(
                    span=proposal.span,
                    source=proposal.source,
                    candidate_types=proposal.candidate_types,
                    accepted=False,
                    reason="unresolved_type",
                )
                for proposal in ambiguous
            ),
            *self._typed_decisions(candidates, selected_keys),
        ]
        return SpanResolutionResult(
            selected=selected,
            decisions=tuple(sorted(decisions, key=_decision_order)),
        )

    def _merge_exact(self, proposals: tuple[EntityProposal, ...]) -> tuple[_Candidate, ...]:
        grouped: dict[
            tuple[tuple[int, int], EntityType],
            list[EntityProposal],
        ] = {}
        for proposal in proposals:
            grouped.setdefault((proposal.span, _resolved_type(proposal)), []).append(proposal)

        candidates: list[_Candidate] = []
        for source_proposals in grouped.values():
            ordered = tuple(sorted(source_proposals, key=_proposal_preference))
            representative = ordered[0]
            sources = tuple(sorted({proposal.source for proposal in ordered}))
            merged = replace(
                representative,
                score=max(proposal.score for proposal in ordered),
                evidence_ids=tuple(
                    sorted(
                        {
                            evidence_id
                            for proposal in ordered
                            for evidence_id in proposal.evidence_ids
                        }
                    )
                ),
                concept_ids=tuple(
                    sorted(
                        {
                            concept_id
                            for proposal in ordered
                            for concept_id in proposal.concept_ids
                        }
                    )
                ),
                features=tuple(
                    sorted(
                        {
                            feature
                            for proposal in ordered
                            for feature in proposal.features
                        }
                    )
                ),
            )
            candidates.append(
                _Candidate(
                    representative=merged,
                    source_proposals=ordered,
                    utility=self.scoring.score(merged, source_count=len(sources)),
                )
            )
        return tuple(sorted(candidates, key=_candidate_end_order))

    @staticmethod
    def _maximum_utility_set(candidates: tuple[_Candidate, ...]) -> tuple[_Candidate, ...]:
        ends = [candidate.representative.span[1] for candidate in candidates]
        best: list[_Selection] = [_Selection()]
        for index, candidate in enumerate(candidates):
            predecessor = bisect_right(
                ends,
                candidate.representative.span[0],
                hi=index,
            )
            included = best[predecessor].append(candidate)
            excluded = best[index]
            best.append(_better_selection(included, excluded))
        return best[-1].candidates

    @staticmethod
    def _typed_decisions(
        candidates: tuple[_Candidate, ...],
        selected_keys: set[tuple[tuple[int, int], EntityType]],
    ) -> list[ProposalDecision]:
        selected_spans = [
            candidate.representative.span
            for candidate in candidates
            if (candidate.representative.span, candidate.representative.entity_type)
            in selected_keys
        ]
        decisions: list[ProposalDecision] = []
        for candidate in candidates:
            proposal_key = (
                candidate.representative.span,
                _resolved_type(candidate.representative),
            )
            accepted = proposal_key in selected_keys
            competing_sources = tuple(
                sorted(
                    {
                        other.representative.source
                        for other in candidates
                        if other is not candidate
                        and _overlaps(
                            candidate.representative.span,
                            other.representative.span,
                        )
                    }
                )
            )
            reason = (
                "selected_exact_consensus"
                if accepted and len(candidate.source_proposals) > 1
                else "selected_global_utility"
                if accepted
                else "rejected_overlap"
                if any(
                    _overlaps(candidate.representative.span, selected_span)
                    for selected_span in selected_spans
                )
                else "rejected_global_utility"
            )
            for proposal in candidate.source_proposals:
                decisions.append(
                    ProposalDecision(
                        span=proposal.span,
                        source=proposal.source,
                        candidate_types=proposal.candidate_types,
                        accepted=accepted,
                        reason=reason,
                        selected_type=proposal.entity_type if accepted else None,
                        competing_sources=competing_sources,
                    )
                )
        return decisions


def _better_selection(left: _Selection, right: _Selection) -> _Selection:
    left_rank = (
        round(left.utility, 12),
        left.source_agreement,
        left.covered_characters,
        -len(left.candidates),
    )
    right_rank = (
        round(right.utility, 12),
        right.source_agreement,
        right.covered_characters,
        -len(right.candidates),
    )
    if left_rank != right_rank:
        return left if left_rank > right_rank else right
    return left if _selection_signature(left) < _selection_signature(right) else right


def _selection_signature(
    selection: _Selection,
) -> tuple[tuple[int, int, str, str], ...]:
    return tuple(_proposal_start_order(candidate.representative) for candidate in selection.candidates)


def _proposal_preference(proposal: EntityProposal) -> tuple[float, int, str]:
    return (-proposal.score, -(proposal.span[1] - proposal.span[0]), proposal.source)


def _candidate_end_order(candidate: _Candidate) -> tuple[int, int, str, str]:
    proposal = candidate.representative
    assert proposal.entity_type is not None
    return (proposal.span[1], proposal.span[0], proposal.entity_type.value, proposal.source)


def _proposal_start_order(proposal: EntityProposal) -> tuple[int, int, str, str]:
    assert proposal.entity_type is not None
    return (proposal.span[0], proposal.span[1], proposal.entity_type.value, proposal.source)


def _decision_order(decision: ProposalDecision) -> tuple[int, int, str, str]:
    return (
        decision.span[0],
        decision.span[1],
        decision.source,
        ",".join(item.value for item in decision.candidate_types),
    )


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _resolved_type(proposal: EntityProposal) -> EntityType:
    entity_type = proposal.entity_type
    if entity_type is None:
        raise ValueError("Span resolver candidate must have one resolved entity type")
    return entity_type
