"""Evidence-weighted composition of model and dictionary entity proposals."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from medical_kg_nlp.schema.annotation import (
    AmbiguousEntityProposal,
    EntityAnnotation,
)
from medical_kg_nlp.schema.types import EntityType

if TYPE_CHECKING:
    from medical_kg_nlp.pipeline.ports import EntityExtractorPort

__all__ = ["HybridArbitrationPolicy", "HybridEntityExtractorAdapter"]


@dataclass(frozen=True)
class HybridArbitrationPolicy:
    """Source priors used to arbitrate overlapping NER proposals.

    The values are intentionally small relative to an extractor confidence. They preserve a
    modest preference for reviewed lexical evidence without making dictionary spans impossible
    for a stronger model proposal to correct.
    """

    dictionary_prior: float = 0.04
    exact_agreement_bonus: float = 0.12
    ambiguous_type_support_bonus: float = 0.04

    def __post_init__(self) -> None:
        for name, value in (
            ("dictionary_prior", self.dictionary_prior),
            ("exact_agreement_bonus", self.exact_agreement_bonus),
            ("ambiguous_type_support_bonus", self.ambiguous_type_support_bonus),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")


@dataclass(frozen=True)
class HybridEntityExtractorAdapter:
    """Resolve model and dictionary proposals without making either source authoritative.

    Exact span/type agreement receives a consensus bonus. Remaining overlap conflicts are
    resolved globally so two supported non-overlapping mentions can beat one broad proposal.
    """

    model: EntityExtractorPort
    dictionary: EntityExtractorPort
    policy: HybridArbitrationPolicy = HybridArbitrationPolicy()

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        ambiguous_proposals: tuple[AmbiguousEntityProposal, ...] = ()
        extract_with_proposals = getattr(
            self.dictionary,
            "extract_with_proposals",
            None,
        )
        if extract_with_proposals is not None:
            result = extract_with_proposals(source_text)
            dictionary_entities = self._validated(list(result.entities), source_text)
            ambiguous_proposals = self._validated_ambiguous(
                result.ambiguous_proposals,
                source_text,
            )
        else:
            dictionary_entities = self._validated(
                self.dictionary.extract(source_text),
                source_text,
            )
        model_entities = self._validated(self.model.extract(source_text), source_text)
        model_entities = _apply_ambiguous_type_support(
            model_entities,
            ambiguous_proposals,
            bonus=self.policy.ambiguous_type_support_bonus,
        )
        proposals = _merge_exact_proposals(
            dictionary_entities,
            model_entities,
            policy=self.policy,
        )
        selected = [proposal.entity for proposal in _maximum_evidence_set(proposals)]
        selected.sort(key=_entity_order)
        # INVARIANT: IDs are regenerated after fusion so downstream relations see unique IDs.
        return [replace(entity, id=f"H{index:04d}") for index, entity in enumerate(selected, 1)]

    def close(self) -> None:
        """Close model and dictionary children when they own expensive resources."""

        _close_child(self.model)
        _close_child(self.dictionary)

    @staticmethod
    def _validated(
        entities: list[EntityAnnotation],
        source_text: str,
    ) -> list[EntityAnnotation]:
        for entity in entities:
            entity.validate_offsets(source_text)
            if not math.isfinite(entity.confidence) or not 0.0 <= entity.confidence <= 1.0:
                raise ValueError(
                    f"Entity {entity.id!r} confidence must be finite and between 0 and 1"
                )
        return entities

    @staticmethod
    def _validated_ambiguous(
        proposals: tuple[AmbiguousEntityProposal, ...],
        source_text: str,
    ) -> tuple[AmbiguousEntityProposal, ...]:
        for proposal in proposals:
            proposal.validate_offsets(source_text)
        return proposals


@dataclass(frozen=True)
class _Proposal:
    entity: EntityAnnotation
    utility: float
    sources: frozenset[str]

    @property
    def has_agreement(self) -> bool:
        return self.sources == frozenset({"dictionary", "model"})


@dataclass(frozen=True)
class _Selection:
    proposals: tuple[_Proposal, ...] = ()
    utility: float = 0.0
    agreement_count: int = 0
    covered_characters: int = 0

    def append(self, proposal: _Proposal) -> "_Selection":
        start, end = proposal.entity.span
        return _Selection(
            proposals=(*self.proposals, proposal),
            utility=self.utility + proposal.utility,
            agreement_count=self.agreement_count + int(proposal.has_agreement),
            covered_characters=self.covered_characters + end - start,
        )


def _merge_exact_proposals(
    dictionary_entities: list[EntityAnnotation],
    model_entities: list[EntityAnnotation],
    *,
    policy: HybridArbitrationPolicy,
) -> list[_Proposal]:
    """Collapse exact span/type duplicates while retaining source agreement evidence."""

    grouped: dict[
        tuple[tuple[int, int], EntityType],
        dict[str, list[EntityAnnotation]],
    ] = {}
    for source, entities in (
        ("dictionary", dictionary_entities),
        ("model", model_entities),
    ):
        for entity in entities:
            grouped.setdefault((entity.span, entity.type), {}).setdefault(source, []).append(entity)

    proposals: list[_Proposal] = []
    for source_entities in grouped.values():
        dictionary_best = _best_entity(source_entities.get("dictionary", []))
        model_best = _best_entity(source_entities.get("model", []))
        source_scores = []
        if dictionary_best is not None:
            source_scores.append(dictionary_best.confidence + policy.dictionary_prior)
        if model_best is not None:
            source_scores.append(model_best.confidence)
        sources = frozenset(source_entities)
        utility = max(source_scores)
        if len(sources) == 2:
            utility += policy.exact_agreement_bonus

        winner = _proposal_winner(
            dictionary_best,
            model_best,
            dictionary_prior=policy.dictionary_prior,
        )
        confidence = max(
            entity.confidence
            for entities in source_entities.values()
            for entity in entities
        )
        medication_mention = next(
            (
                entity.medication_mention
                for entities in source_entities.values()
                for entity in entities
                if entity.medication_mention is not None
            ),
            winner.medication_mention,
        )
        proposals.append(
            _Proposal(
                entity=replace(
                    winner,
                    confidence=confidence,
                    medication_mention=medication_mention,
                ),
                utility=utility,
                sources=sources,
            )
        )
    return proposals


def _best_entity(entities: list[EntityAnnotation]) -> EntityAnnotation | None:
    if not entities:
        return None
    return min(
        entities,
        key=lambda entity: (
            -entity.confidence,
            -(entity.span[1] - entity.span[0]),
            entity.id,
        ),
    )


def _apply_ambiguous_type_support(
    entities: list[EntityAnnotation],
    proposals: tuple[AmbiguousEntityProposal, ...],
    *,
    bonus: float,
) -> list[EntityAnnotation]:
    """Reward model types supported by an unresolved exact dictionary span.

    The ambiguous proposal never becomes a final entity on its own. This keeps rule-only behavior
    precision-first while allowing an independent model to resolve the type without losing lexical
    evidence.
    """

    supported = {
        (proposal.span, entity_type)
        for proposal in proposals
        for entity_type in proposal.candidate_types
    }
    return [
        replace(entity, confidence=min(1.0, entity.confidence + bonus))
        if (entity.span, entity.type) in supported
        else entity
        for entity in entities
    ]


def _proposal_winner(
    dictionary: EntityAnnotation | None,
    model: EntityAnnotation | None,
    *,
    dictionary_prior: float,
) -> EntityAnnotation:
    if dictionary is None:
        assert model is not None
        return model
    if model is None:
        return dictionary
    dictionary_score = dictionary.confidence + dictionary_prior
    if dictionary_score >= model.confidence:
        return dictionary
    return model


def _maximum_evidence_set(proposals: list[_Proposal]) -> tuple[_Proposal, ...]:
    """Select a deterministic maximum-weight set of non-overlapping raw spans.

    SCALING: weighted interval scheduling avoids pairwise greedy suppression and remains
    ``O(n log n)`` for long notes with many model and terminology proposals.
    """

    ordered = sorted(proposals, key=_proposal_end_order)
    ends = [proposal.entity.span[1] for proposal in ordered]
    best: list[_Selection] = [_Selection()]
    for index, proposal in enumerate(ordered):
        predecessor = bisect_right(ends, proposal.entity.span[0], hi=index)
        included = best[predecessor].append(proposal)
        excluded = best[index]
        best.append(_better_selection(included, excluded))
    return best[-1].proposals


def _better_selection(left: _Selection, right: _Selection) -> _Selection:
    left_rank = (
        round(left.utility, 12),
        left.agreement_count,
        left.covered_characters,
        -len(left.proposals),
    )
    right_rank = (
        round(right.utility, 12),
        right.agreement_count,
        right.covered_characters,
        -len(right.proposals),
    )
    if left_rank != right_rank:
        return left if left_rank > right_rank else right
    return left if _selection_signature(left) < _selection_signature(right) else right


def _selection_signature(selection: _Selection) -> tuple[tuple[int, int, str, str], ...]:
    return tuple(_entity_order(proposal.entity) for proposal in selection.proposals)


def _proposal_end_order(proposal: _Proposal) -> tuple[int, int, str, str]:
    entity = proposal.entity
    return (entity.span[1], entity.span[0], entity.type.value, entity.id)


def _entity_order(entity: EntityAnnotation) -> tuple[int, int, str, str]:
    return (entity.span[0], entity.span[1], entity.type.value, entity.id)


def _close_child(child: object) -> None:
    close = getattr(child, "close", None)
    if callable(close):
        close()
