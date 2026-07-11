from __future__ import annotations

import re
from dataclasses import replace

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match, token_set


_STRENGTH_RE = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:mcg|µg|μg|mg|g|kg|ml|l|meq|iu|u)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_FORM_CUES = {
    "capsule": frozenset({"capsule", "viên nang"}),
    "inhaler": frozenset({"inhaler", "hít", "nebs", "nebulizer", "khí dung"}),
    "injection": frozenset({"injection", "injectable", "iv", "tiêm", "tĩnh mạch"}),
    "solution": frozenset({"solution", "syrup", "dung dịch", "dịch"}),
    "tablet": frozenset({"tablet", "viên", "po", "oral", "uống"}),
}


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
        candidate_strengths = _strengths(candidate_name_text)
        explicit_candidate_forms = _forms(candidate_name_text)
        candidate_forms = _forms(f"{candidate_name_text} {entry.dose_form or ''}")

        score = candidate.score
        if mention_strengths and candidate_strengths:
            if mention_strengths.isdisjoint(candidate_strengths):
                return max(0.0, score * 0.2)
            score += 0.10
        if mention_forms and candidate_forms:
            if not mention_forms.isdisjoint(candidate_forms):
                score += 0.06
            elif explicit_candidate_forms:
                score -= 0.12

        normalized_mention = normalize_for_match(mention)
        if entry.semantic_type == EntityType.DRUG and entry.ingredient:
            ingredient = normalize_for_match(entry.ingredient)
            if ingredient and ingredient in normalized_mention:
                score += 0.06

        candidate_tokens = token_set(candidate.canonical_name)
        if candidate_tokens and context_tokens:
            overlap = len(candidate_tokens & context_tokens) / len(candidate_tokens)
            score += min(0.04, 0.04 * overlap)
        return min(1.0, max(0.0, score))


def _strengths(text: str) -> frozenset[str]:
    return frozenset(
        re.sub(r"\s+", "", match.group(0)).replace(",", ".").casefold()
        for match in _STRENGTH_RE.finditer(text)
    )


def _forms(text: str) -> frozenset[str]:
    normalized = normalize_for_match(text)
    return frozenset(
        form
        for form, cues in _FORM_CUES.items()
        if any(re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", normalized) for cue in cues)
    )
