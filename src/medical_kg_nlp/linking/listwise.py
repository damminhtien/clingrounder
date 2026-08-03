"""Model-neutral contracts for bounded cross-candidate reranking.

Listwise models may compare close concepts jointly, but they never own retrieval or create codes.
Every option in these records originates from a type-constrained terminology repository.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Sequence

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.rxnorm_reranker import build_rxnorm_mention_profile
from medical_kg_nlp.retrieval.constraints import ALLOWED_CODE_SYSTEMS
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository

__all__ = [
    "ListwiseCandidateOption",
    "ListwiseCandidateOrder",
    "ListwiseLinkingQuery",
    "ListwiseOrderRanking",
    "ListwiseRerankDecision",
    "ListwiseStructuredMention",
    "aggregate_listwise_rankings",
    "build_listwise_candidate_orders",
    "build_listwise_linking_query",
    "render_listwise_candidate",
]

_MAX_OPTION_COUNT = 26
_MAX_RENDERED_ALIASES = 8


@dataclass(frozen=True, slots=True)
class ListwiseStructuredMention:
    """Normalized attributes that a reranker may compare without changing the raw mention."""

    ingredients: tuple[str, ...] = ()
    brands: tuple[str, ...] = ()
    product_strengths: tuple[str, ...] = ()
    administered_doses: tuple[str, ...] = ()
    ambiguous_strengths: tuple[str, ...] = ()
    dose_forms: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    release_types: tuple[str, ...] = ()

    def to_json(self) -> dict[str, list[str]]:
        return {
            "ingredient": list(self.ingredients),
            "brand": list(self.brands),
            "product_strength": list(self.product_strengths),
            "administered_dose": list(self.administered_doses),
            "ambiguous_strength": list(self.ambiguous_strengths),
            "dose_form": list(self.dose_forms),
            "route": list(self.routes),
            "release": list(self.release_types),
        }


@dataclass(frozen=True, slots=True)
class ListwiseCandidateOption:
    """One retrieved option with bounded terminology metadata."""

    concept_id: str
    code: str | None
    code_system: str
    canonical_name: str
    matched_alias: str | None
    retrieval_score: float
    sources: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    structured_attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.concept_id or not self.code_system or not self.canonical_name:
            raise ValueError("Listwise candidates require concept identity and a title.")
        if not 0.0 <= self.retrieval_score <= 1.0:
            raise ValueError("Listwise retrieval_score must be between 0 and 1.")

    @property
    def identity(self) -> tuple[str, str | None, str]:
        return self.code_system, self.code, self.concept_id

    def to_json(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "code": self.code,
            "code_system": self.code_system,
            "canonical_name": self.canonical_name,
            "matched_alias": self.matched_alias,
            "retrieval_score": self.retrieval_score,
            "sources": list(self.sources),
            "aliases": list(self.aliases),
            "parents": list(self.parents),
            "structured_attributes": dict(self.structured_attributes),
        }


@dataclass(frozen=True, slots=True)
class ListwiseLinkingQuery:
    """One inference query containing only dictionary-constrained candidates."""

    query_id: str
    mention: str
    context: str
    entity_type: EntityType
    structured_mention: ListwiseStructuredMention
    candidates: tuple[ListwiseCandidateOption, ...]

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.mention.strip():
            raise ValueError("Listwise queries require query_id and mention.")
        if not 2 <= len(self.candidates) <= _MAX_OPTION_COUNT:
            raise ValueError("Listwise queries require between 2 and 26 candidates.")
        identities = [candidate.identity for candidate in self.candidates]
        if len(identities) != len(set(identities)):
            raise ValueError("Listwise candidate identities must be unique.")

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "mention": self.mention,
            "context": self.context,
            "entity_type": self.entity_type.value,
            "structured_mention": self.structured_mention.to_json(),
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class ListwiseCandidateOrder:
    """One deterministic presentation order over global candidate indices."""

    order_id: str
    candidate_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ListwiseOrderRanking:
    """One validated model response projected back to global candidate indices."""

    order_id: str
    ranked_candidate_indices: tuple[int, ...]
    abstain: bool
    valid: bool = True


@dataclass(frozen=True, slots=True)
class ListwiseRerankDecision:
    """Aggregated rank evidence from retrieval, reverse, and shuffled presentations."""

    ranked_candidate_indices: tuple[int, ...]
    aggregate_scores: tuple[float, ...]
    abstain: bool
    order_consistency: float
    order_rankings: tuple[ListwiseOrderRanking, ...]


def build_listwise_linking_query(
    *,
    query_id: str,
    mention: str,
    context: str,
    entity_type: EntityType,
    candidates: Sequence[Candidate],
    repository: TerminologyRepository | None = None,
) -> ListwiseLinkingQuery:
    """Build an inference query while enforcing entity/code-system compatibility."""

    entries: list[ConceptEntry] = []
    options: list[ListwiseCandidateOption] = []
    allowed = ALLOWED_CODE_SYSTEMS.get(entity_type)
    for candidate in candidates:
        if candidate.semantic_type is not entity_type:
            raise ValueError(
                f"Candidate {candidate.concept_id} has incompatible semantic type."
            )
        if allowed is not None and candidate.code_system not in allowed:
            raise ValueError(f"Candidate {candidate.concept_id} has incompatible code system.")
        entry = (
            repository.get_by_concept_id(candidate.concept_id)
            if repository is not None
            else None
        )
        if entry is not None:
            _validate_entry(candidate, entry)
            entries.append(entry)
        options.append(_candidate_option(candidate, entry))

    structured = ListwiseStructuredMention()
    if entity_type is EntityType.DRUG:
        profile = build_rxnorm_mention_profile(mention, entries)
        structure = profile.structure
        structured = ListwiseStructuredMention(
            ingredients=tuple(sorted(profile.ingredients)),
            brands=tuple(sorted(profile.brands)),
            product_strengths=tuple(sorted(structure.product_strengths)),
            administered_doses=tuple(sorted(structure.administered_doses)),
            ambiguous_strengths=tuple(sorted(structure.ambiguous_strengths)),
            dose_forms=tuple(sorted(structure.dose_forms)),
            routes=tuple(sorted(structure.routes)),
            release_types=tuple(sorted(structure.release_types)),
        )
    return ListwiseLinkingQuery(
        query_id=query_id,
        mention=mention,
        context=context,
        entity_type=entity_type,
        structured_mention=structured,
        candidates=tuple(options),
    )


def build_listwise_candidate_orders(
    query: ListwiseLinkingQuery,
    *,
    seed: int,
) -> tuple[ListwiseCandidateOrder, ...]:
    """Return retrieval, reverse, and query-stable shuffled presentation orders."""

    retrieval = tuple(range(len(query.candidates)))
    shuffled = list(retrieval)
    derived_seed = int.from_bytes(
        hashlib.sha256(f"{seed}\0{query.query_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    random.Random(derived_seed).shuffle(shuffled)
    return (
        ListwiseCandidateOrder("retrieval", retrieval),
        ListwiseCandidateOrder("reverse", tuple(reversed(retrieval))),
        ListwiseCandidateOrder("shuffled", tuple(shuffled)),
    )


def aggregate_listwise_rankings(
    query: ListwiseLinkingQuery,
    rankings: Sequence[ListwiseOrderRanking],
) -> ListwiseRerankDecision:
    """Aggregate valid non-abstaining orders with mean normalized rank."""

    if not rankings:
        raise ValueError("At least one listwise order ranking is required.")
    candidate_count = len(query.candidates)
    expected = set(range(candidate_count))
    valid = [ranking for ranking in rankings if ranking.valid]
    for ranking in valid:
        if set(ranking.ranked_candidate_indices) != expected or len(
            ranking.ranked_candidate_indices
        ) != candidate_count:
            raise ValueError(f"Ranking {ranking.order_id} is not a candidate permutation.")
    active = [ranking for ranking in valid if not ranking.abstain]
    abstain_votes = sum(ranking.abstain for ranking in valid) + (len(rankings) - len(valid))
    abstain = not active or abstain_votes * 2 >= len(rankings) + 1
    score_sums = [0.0] * candidate_count
    if active:
        for ranking in active:
            for rank, candidate_index in enumerate(ranking.ranked_candidate_indices):
                score_sums[candidate_index] += 1.0 - rank / candidate_count
        aggregate_scores = tuple(value / len(active) for value in score_sums)
        ranked_indices = tuple(
            sorted(range(candidate_count), key=lambda index: (-aggregate_scores[index], index))
        )
        top_counts: dict[int, int] = {}
        for ranking in active:
            top = ranking.ranked_candidate_indices[0]
            top_counts[top] = top_counts.get(top, 0) + 1
        order_consistency = max(top_counts.values()) / len(active)
    else:
        aggregate_scores = tuple(0.0 for _ in range(candidate_count))
        ranked_indices = tuple(range(candidate_count))
        order_consistency = 0.0
    return ListwiseRerankDecision(
        ranked_candidate_indices=ranked_indices,
        aggregate_scores=aggregate_scores,
        abstain=abstain,
        order_consistency=order_consistency,
        order_rankings=tuple(rankings),
    )


def render_listwise_candidate(candidate: ListwiseCandidateOption) -> str:
    """Render bounded candidate metadata without exposing retrieval implementation details."""

    aliases = "; ".join(candidate.aliases) or "-"
    parents = "; ".join(candidate.parents) or "-"
    attributes = "; ".join(
        f"{key}={value}" for key, value in candidate.structured_attributes
    ) or "-"
    return (
        f"{candidate.code_system}:{candidate.code or candidate.concept_id} | "
        f"title={candidate.canonical_name} | aliases={aliases} | "
        f"parent={parents} | attributes={attributes}"
    )


def _candidate_option(
    candidate: Candidate,
    entry: ConceptEntry | None,
) -> ListwiseCandidateOption:
    aliases: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()
    if entry is not None:
        excluded = {candidate.canonical_name, candidate.matched_alias}
        aliases = tuple(
            name
            for name in entry.all_names
            if name not in excluded
        )[:_MAX_RENDERED_ALIASES]
        parents = tuple(dict.fromkeys((*entry.parents, entry.parent_code))) if entry.parent_code else entry.parents
        attributes = tuple(
            (key, value)
            for key, value in (
                ("ingredient", entry.ingredient),
                ("brand", entry.brand_name),
                ("strength", entry.strength),
                ("dose_form", entry.dose_form),
                ("tty", entry.rxnorm_tty),
            )
            if value
        )
    return ListwiseCandidateOption(
        concept_id=candidate.concept_id,
        code=candidate.code,
        code_system=candidate.code_system.value,
        canonical_name=candidate.canonical_name,
        matched_alias=candidate.matched_alias,
        retrieval_score=candidate.score,
        sources=candidate.sources,
        aliases=aliases,
        parents=parents,
        structured_attributes=attributes,
    )


def _validate_entry(candidate: Candidate, entry: ConceptEntry) -> None:
    if (
        entry.concept_id != candidate.concept_id
        or entry.code != candidate.code
        or entry.code_system is not candidate.code_system
        or entry.semantic_type is not candidate.semantic_type
    ):
        raise ValueError(
            f"Terminology entry does not match candidate {candidate.concept_id}."
        )
