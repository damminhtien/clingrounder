"""Structured, dictionary-constrained RxNorm reranking for medication mentions.

The retriever remains responsible for recall. This reranker only compares a bounded candidate
set with medication identity and product evidence already present in the raw mention.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.linking.candidate import Candidate, CandidateEvidence
from medical_kg_nlp.linking.reranker import HeuristicReranker
from medical_kg_nlp.linking.structured_rxnorm import (
    RxNormCompatibility,
    RxNormMentionProfile,
    parse_rxnorm_entry_structure,
    rxnorm_compatibility,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "StructuredRxNormReranker",
    "StructuredRxNormScore",
    "build_rxnorm_mention_profile",
]

_BARE_TTYS = frozenset({"IN", "MIN", "PIN", "BN"})
_PRODUCT_TTYS = frozenset({"SCD", "SBD"})
_SOURCE_PRIORS = {
    "reviewed_memory": 1.0,
    "exact": 1.0,
    "toneless": 0.92,
    "abbreviation": 0.90,
    "dense": 0.70,
    "fuzzy": 0.65,
    "char_ngram": 0.55,
    "bm25": 0.50,
}
_BONUS_BY_REASON = {
    "ingredient_exact": 0.30,
    "brand_exact": 0.15,
    "product_strength_exact": 0.30,
    "dose_form_exact": 0.15,
    "release_exact": 0.20,
    # An administered dose is never a product-strength assertion. It may only break a tie.
    "administered_dose_match": 0.02,
}
_PENALTY_BY_REASON = {
    "brand_mismatch": 0.15,
    "brand_evidence_missing": 0.10,
    "ambiguous_strength": 0.05,
    "route_only_discrepancy": 0.02,
    "administered_dose_discrepancy": 0.02,
}
_IDENTITY_SPLIT_RE = re.compile(r"\s*(?:/|\+|\band\b|\bvà\b)\s*", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class StructuredRxNormScore:
    """Auditable components used to rank one RxNorm candidate.

    ``final_score`` is bounded to the linker score domain. ``reasons`` contain every applied
    compatibility, TTY, evidence-fallback, and soft-dose decision so production traces can be
    inspected without recomputing a score.
    """

    lexical_score: float
    dense_score: float
    identity_score: float
    structure_score: float
    source_prior: float
    final_score: float
    hard_conflict: str | None
    reasons: tuple[str, ...]


class StructuredRxNormReranker:
    """Rank RxNorm medications with explicit composition and product evidence.

    Non-RxNorm candidates retain the generic heuristic behavior. For RxNorm, an explicit
    ingredient, release, product strength, or dose-form conflict removes the candidate before
    qualification. Route and administered-dose evidence can influence ranking but never hard
    reject a manufactured product.
    """

    def __init__(self, repository: TerminologyRepository) -> None:
        self.repository = repository
        self.fallback = HeuristicReranker(repository)

    def rerank(
        self,
        candidates: list[Candidate],
        context_window: str = "",
        mention: str = "",
    ) -> list[Candidate]:
        """Filter hard conflicts and deterministically rank the remaining candidates."""

        entries = {
            candidate.concept_id: entry
            for candidate in candidates
            if (entry := self.repository.get_by_concept_id(candidate.concept_id)) is not None
        }
        profile = build_rxnorm_mention_profile(mention, entries.values())
        reranked: list[Candidate] = []
        fallback_candidates: list[Candidate] = []
        for candidate in candidates:
            entry = entries.get(candidate.concept_id)
            if entry is None or not _is_rxnorm_drug(candidate, entry):
                fallback_candidates.append(candidate)
                continue
            score = self.score_candidate(
                candidate,
                entry,
                mention=mention,
                profile=profile,
            )
            if score.hard_conflict is not None:
                continue
            reranked.append(replace(candidate, score=score.final_score))

        # Non-RxNorm ranking remains byte-for-byte owned by the previous generic strategy.
        reranked.extend(
            self.fallback.rerank(
                fallback_candidates,
                context_window=context_window,
                mention=mention,
            )
        )
        return sorted(reranked, key=_candidate_order)

    def score_candidate(
        self,
        candidate: Candidate,
        entry: ConceptEntry,
        *,
        mention: str,
        profile: RxNormMentionProfile | None = None,
    ) -> StructuredRxNormScore:
        """Return the complete structured score for one dictionary-constrained candidate."""

        active_profile = profile or build_rxnorm_mention_profile(mention, (entry,))
        compatibility = rxnorm_compatibility(
            mention,
            entry,
            ingredients=active_profile.ingredients,
            brands=active_profile.brands,
        )
        lexical_score, dense_score, source_prior, evidence_reasons = _retrieval_evidence(
            candidate
        )
        identity_score = _identity_score(active_profile, compatibility)
        structure_score, structure_reasons = _structure_score(
            active_profile,
            entry,
        )
        reasons = [*compatibility.bonuses, *compatibility.penalties]
        reasons.extend(evidence_reasons)
        reasons.extend(structure_reasons)
        if compatibility.hard_conflict is not None:
            return StructuredRxNormScore(
                lexical_score=lexical_score,
                dense_score=dense_score,
                identity_score=identity_score,
                structure_score=structure_score,
                source_prior=source_prior,
                final_score=0.0,
                hard_conflict=compatibility.hard_conflict,
                reasons=tuple(sorted(set(reasons))),
            )

        # MODEL: lexical and dense evidence are independently weighted when available. The
        # current fused Candidate record has no separate dense score in lexical-only pipelines,
        # so its bounded retrieval score is intentionally reused as a neutral fallback.
        score = (
            0.35 * lexical_score
            + 0.20 * dense_score
            + 0.20 * identity_score
            + 0.15 * structure_score
            + 0.10 * source_prior
        )
        score += sum(_BONUS_BY_REASON.get(reason, 0.0) for reason in reasons)
        score -= sum(_PENALTY_BY_REASON.get(reason, 0.0) for reason in reasons)
        return StructuredRxNormScore(
            lexical_score=lexical_score,
            dense_score=dense_score,
            identity_score=identity_score,
            structure_score=structure_score,
            source_prior=source_prior,
            final_score=_bounded_score(score),
            hard_conflict=None,
            reasons=tuple(sorted(set(reasons))),
        )


def build_rxnorm_mention_profile(
    mention: str,
    entries: Iterable[ConceptEntry],
) -> RxNormMentionProfile:
    """Extract only identity strings present in both the raw mention and retrieved RxNorm rows."""

    ingredients: set[str] = set()
    brands: set[str] = set()
    for entry in entries:
        if entry.ingredient is not None:
            for ingredient in _IDENTITY_SPLIT_RE.split(entry.ingredient):
                if _contains_identity(mention, ingredient):
                    ingredients.add(ingredient)
        if entry.brand_name is not None and _contains_identity(mention, entry.brand_name):
            brands.add(entry.brand_name)
    return RxNormMentionProfile.from_text(
        mention,
        ingredients=ingredients,
        brands=brands,
    )


def _contains_identity(mention: str, value: str) -> bool:
    normalized_mention = normalize_for_match(mention)
    normalized_value = normalize_for_match(value)
    if not normalized_value:
        return False
    return (
        re.search(
            rf"(?<!\w){re.escape(normalized_value)}(?!\w)",
            normalized_mention,
        )
        is not None
    )


def _is_rxnorm_drug(candidate: Candidate, entry: ConceptEntry | None) -> bool:
    return (
        entry is not None
        and candidate.code_system is CodeSystem.RXNORM
        and candidate.semantic_type is EntityType.DRUG
        and entry.semantic_type is EntityType.DRUG
    )


def _retrieval_evidence(candidate: Candidate) -> tuple[float, float, float, tuple[str, ...]]:
    evidence = candidate.evidence or (
        CandidateEvidence(
            source=candidate.source,
            score=candidate.score,
            rank=1,
            concept_id=candidate.concept_id,
            matched_alias=candidate.matched_alias,
        ),
    )
    dense_scores = [item.score for item in evidence if item.source == "dense"]
    lexical_scores = [item.score for item in evidence if item.source != "dense"]
    lexical_score = max(lexical_scores, default=candidate.score)
    reasons: list[str] = []
    if dense_scores:
        dense_score = max(dense_scores)
    else:
        dense_score = lexical_score
        reasons.append("dense_score_fallback_to_retrieval")
    source_prior = max(
        (_SOURCE_PRIORS.get(item.source, 0.40) for item in evidence),
        default=0.40,
    )
    return lexical_score, dense_score, source_prior, tuple(reasons)


def _identity_score(
    profile: RxNormMentionProfile,
    compatibility: RxNormCompatibility,
) -> float:
    if "ingredient_exact" in compatibility.bonuses:
        return 1.0
    if "brand_exact" in compatibility.bonuses:
        return 0.75
    if profile.ingredients or profile.brands:
        return 0.25
    return 0.50


def _structure_score(
    profile: RxNormMentionProfile,
    entry: ConceptEntry,
) -> tuple[float, tuple[str, ...]]:
    mention_structure = profile.structure
    candidate_structure = parse_rxnorm_entry_structure(entry)
    reasons: list[str] = []
    if not mention_structure.has_product_evidence:
        if entry.rxnorm_tty in _BARE_TTYS:
            reasons.append("tty_prefer_bare")
            score = 0.80
        else:
            reasons.append("tty_deprioritize_product_without_evidence")
            score = 0.30
    elif entry.rxnorm_tty in _PRODUCT_TTYS:
        reasons.append("tty_prefer_structured_product")
        score = 1.0
    else:
        reasons.append("tty_deprioritize_bare_with_product_evidence")
        score = 0.35

    if mention_structure.ambiguous_strengths:
        reasons.append("ambiguous_strength")
    if mention_structure.routes and candidate_structure.routes:
        if mention_structure.routes.isdisjoint(candidate_structure.routes):
            # Route is administration context, not an RxNorm dose form. It can only nudge rank.
            reasons.append("route_only_discrepancy")
    if mention_structure.administered_doses and candidate_structure.product_strengths:
        if mention_structure.administered_doses == candidate_structure.product_strengths:
            reasons.append("administered_dose_match")
        elif mention_structure.administered_doses.isdisjoint(
            candidate_structure.product_strengths
        ):
            reasons.append("administered_dose_discrepancy")
    return score, tuple(reasons)


def _candidate_order(candidate: Candidate) -> tuple[float, str, str, str]:
    return (
        -candidate.score,
        candidate.code_system.value,
        candidate.code or "",
        candidate.concept_id,
    )


def _bounded_score(value: float) -> float:
    return min(1.0, max(0.0, value))
