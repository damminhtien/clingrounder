from __future__ import annotations

from dataclasses import replace
import re

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.structured_rxnorm import parse_medication_structure
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match, token_set


_STRUCTURED_PRODUCT_TTYS = frozenset({"SCD", "SBD", "SCDF", "SBDF", "GPCK", "BPCK"})
_INGREDIENT_TTYS = frozenset({"IN", "PIN", "MIN"})
_ADMINISTERED_SIG_RE = re.compile(
    r"(?<!\w)(?:po|p\.o\.|iv|im|sc|sl|bid|tid|qid|qhs|qam|qd|daily|prn|"
    r"uống|tiêm|truyền|mỗi\s+ngày|hằng\s+ngày)(?!\w)|\d\s*-\s*\d",
    flags=re.IGNORECASE | re.UNICODE,
)


class HeuristicReranker:
    def __init__(self, store: DictionaryStore) -> None:
        self.store = store

    def rerank(
        self,
        candidates: list[Candidate],
        context_window: str = "",
        mention: str = "",
    ) -> list[Candidate]:
        mention_strengths = _strengths(mention)
        mention_forms = _forms(mention)
        context_tokens = token_set(context_window)
        reranked = [
            replace(
                candidate,
                score=self._candidate_score(
                    candidate,
                    mention=mention,
                    mention_strengths=mention_strengths,
                    mention_forms=mention_forms,
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
        mention_strengths: frozenset[str],
        mention_forms: frozenset[str],
        context_tokens: set[str],
    ) -> float:
        entry = self.store.by_concept_id.get(candidate.concept_id)
        if entry is None:
            return candidate.score
        candidate_name_text = " ".join(entry.all_names)
        candidate_strengths = _strengths(f"{candidate_name_text} {entry.strength or ''}")
        explicit_candidate_forms = _forms(candidate_name_text)
        candidate_forms = _forms(f"{candidate_name_text} {entry.dose_form or ''}")

        score = candidate.score
        if mention_strengths and candidate_strengths:
            if not mention_strengths.isdisjoint(candidate_strengths):
                score += 0.10
            elif not _ADMINISTERED_SIG_RE.search(mention):
                return max(0.0, score * 0.2)
        elif entry.rxnorm_tty in _STRUCTURED_PRODUCT_TTYS and candidate_strengths:
            score -= 0.24
        elif mention_strengths and entry.rxnorm_tty in _INGREDIENT_TTYS:
            score -= 0.08
        if mention_forms and candidate_forms:
            if not mention_forms.isdisjoint(candidate_forms):
                score += 0.06
            elif explicit_candidate_forms:
                score -= 0.12
        elif entry.rxnorm_tty in _STRUCTURED_PRODUCT_TTYS and explicit_candidate_forms:
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

        candidate_tokens = token_set(candidate.canonical_name)
        if candidate_tokens and context_tokens:
            overlap = len(candidate_tokens & context_tokens) / len(candidate_tokens)
            score += min(0.04, 0.04 * overlap)
        return min(1.0, max(0.0, score))


def _strengths(text: str) -> frozenset[str]:
    return parse_medication_structure(text).strengths


def _forms(text: str) -> frozenset[str]:
    return parse_medication_structure(text).dose_forms
