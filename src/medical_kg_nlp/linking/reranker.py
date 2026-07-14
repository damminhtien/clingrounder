from __future__ import annotations

from dataclasses import replace

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.structured_rxnorm import (
    MedicationStructure,
    parse_medication_structure,
    parse_rxnorm_entry_structure,
)
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match, token_set


_STRUCTURED_PRODUCT_TTYS = frozenset({"SCD", "SBD", "SCDF", "SBDF", "GPCK", "BPCK"})
_INGREDIENT_TTYS = frozenset({"IN", "PIN", "MIN"})
_BRAND_TTYS = frozenset({"BN"})


class HeuristicReranker:
    def __init__(self, store: DictionaryStore) -> None:
        self.store = store

    def rerank(
        self,
        candidates: list[Candidate],
        context_window: str = "",
        mention: str = "",
    ) -> list[Candidate]:
        mention_structure = parse_medication_structure(mention)
        context_tokens = token_set(context_window)
        reranked = [
            replace(
                candidate,
                score=self._candidate_score(
                    candidate,
                    mention=mention,
                    mention_structure=mention_structure,
                    context_tokens=context_tokens,
                ),
            )
            for candidate in candidates
        ]
        return sorted(
            reranked,
            key=lambda candidate: (
                -candidate.score,
                candidate.code_system.value,
                candidate.code or "",
                candidate.concept_id,
            ),
        )

    def _candidate_score(
        self,
        candidate: Candidate,
        *,
        mention: str,
        mention_structure: MedicationStructure,
        context_tokens: set[str],
    ) -> float:
        entry = self.store.by_concept_id.get(candidate.concept_id)
        if entry is None:
            return candidate.score
        candidate_structure = parse_rxnorm_entry_structure(entry)

        score = candidate.score
        candidate_strengths = candidate_structure.product_strengths
        if mention_structure.product_strengths and candidate_strengths:
            if mention_structure.product_strengths.isdisjoint(candidate_strengths):
                return max(0.0, score * 0.1)
        elif mention_structure.ambiguous_strengths and candidate_strengths:
            # Ambiguous SIG strength may describe an administered dose. Prefer a matching product,
            # but do not apply the hard conflict reserved for explicit product evidence.
            if mention_structure.ambiguous_strengths.isdisjoint(candidate_strengths):
                score *= 0.7
        elif entry.rxnorm_tty in _STRUCTURED_PRODUCT_TTYS and candidate_strengths:
            score -= 0.24
        elif mention_structure.has_product_evidence and entry.rxnorm_tty in _INGREDIENT_TTYS:
            score -= 0.08
        if mention_structure.dose_forms and candidate_structure.dose_forms:
            if mention_structure.dose_forms.isdisjoint(candidate_structure.dose_forms):
                score -= 0.12
        elif entry.rxnorm_tty in _STRUCTURED_PRODUCT_TTYS and candidate_structure.dose_forms:
            score -= 0.08

        normalized_mention = normalize_for_match(mention)
        if entry.semantic_type == EntityType.DRUG and entry.ingredient:
            ingredient = normalize_for_match(entry.ingredient)
            if ingredient and ingredient in normalized_mention:
                score += 0.06
        if entry.semantic_type == EntityType.DRUG and entry.brand_name:
            brand = normalize_for_match(entry.brand_name)
            if brand and brand in normalized_mention:
                score += 0.04

        if not mention_structure.has_product_evidence:
            if entry.rxnorm_tty in _STRUCTURED_PRODUCT_TTYS:
                score -= 0.08
        elif entry.rxnorm_tty in {*_INGREDIENT_TTYS, *_BRAND_TTYS}:
            score -= 0.08

        candidate_tokens = token_set(candidate.canonical_name)
        if candidate_tokens and context_tokens:
            overlap = len(candidate_tokens & context_tokens) / len(candidate_tokens)
            score += min(0.04, 0.04 * overlap)
        return min(1.0, max(0.0, score))
