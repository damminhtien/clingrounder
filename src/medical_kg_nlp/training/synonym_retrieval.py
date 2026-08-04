"""Hard-negative examples for synonym-aligned terminology encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.training.terminology_pairs import TerminologySynonymPair
from medical_kg_nlp.utils.text import normalize_for_match, token_set

__all__ = ["SynonymRetrievalExample", "build_synonym_retrieval_examples"]


@dataclass(frozen=True, slots=True)
class SynonymRetrievalExample:
    """One positive synonym pair plus hard candidates from other concepts."""

    concept_id: str
    entity_type: EntityType
    query: str
    positive: str
    hard_negatives: tuple[str, ...]


def build_synonym_retrieval_examples(
    pairs: Iterable[TerminologySynonymPair],
    entries: Iterable[ConceptEntry],
    *,
    maximum_hard_negatives: int = 8,
) -> tuple[SynonymRetrievalExample, ...]:
    """Mine deterministic lexical, hierarchy, and medication-structure negatives."""

    if maximum_hard_negatives < 1:
        raise ValueError("maximum_hard_negatives must be positive")
    concepts = tuple(entries)
    by_id = {entry.concept_id: entry for entry in concepts}
    output: list[SynonymRetrievalExample] = []
    for pair in pairs:
        positive_entry = by_id.get(pair.concept_id)
        if positive_entry is None:
            raise ValueError(f"Missing terminology concept for pair {pair.pair_id}")
        candidates: list[tuple[float, str, str]] = []
        for candidate in concepts:
            if candidate.concept_id == positive_entry.concept_id:
                continue
            score = _negative_score(pair.left, positive_entry, candidate)
            if score <= 0.0:
                continue
            candidates.append((score, candidate.canonical_name, candidate.concept_id))
        hard_negatives = tuple(
            name
            for _, name, _ in sorted(
                candidates,
                key=lambda item: (-item[0], normalize_for_match(item[1]), item[2]),
            )[:maximum_hard_negatives]
        )
        output.append(
            SynonymRetrievalExample(
                concept_id=pair.concept_id,
                entity_type=pair.entity_type,
                query=pair.left,
                positive=pair.right,
                hard_negatives=hard_negatives,
            )
        )
    return tuple(output)


def _negative_score(
    query: str,
    positive: ConceptEntry,
    candidate: ConceptEntry,
) -> float:
    query_tokens = token_set(query)
    candidate_tokens = token_set(candidate.canonical_name)
    union = query_tokens | candidate_tokens
    lexical = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
    score = 3.0 * lexical
    if candidate.semantic_type is positive.semantic_type:
        score += 0.5
    elif lexical > 0.0 and {
        candidate.semantic_type,
        positive.semantic_type,
    } <= {EntityType.DISEASE, EntityType.SYMPTOM}:
        score += 1.0
    if positive.parent_code and positive.parent_code == candidate.code:
        score += 4.0
    if candidate.parent_code and candidate.parent_code == positive.code:
        score += 4.0
    if positive.semantic_type is EntityType.DRUG and candidate.semantic_type is EntityType.DRUG:
        positive_ingredient = normalize_for_match(positive.ingredient or "")
        candidate_ingredient = normalize_for_match(candidate.ingredient or "")
        if positive_ingredient and positive_ingredient == candidate_ingredient:
            score += 4.0
            if positive.strength != candidate.strength:
                score += 1.0
        elif positive.dose_form and positive.dose_form == candidate.dose_form:
            score += 2.0
    return score
