from __future__ import annotations
from difflib import SequenceMatcher

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match, token_set


class FuzzyMatcher:
    def __init__(self, store: DictionaryStore, min_score: float = 0.72) -> None:
        self.store = store
        self.min_score = min_score

    def retrieve(self, mention: str, entity_type: EntityType | None = None, limit: int = 20) -> list[Candidate]:
        mention_norm = normalize_for_match(mention, strip_diacritics=True)
        mention_tokens = token_set(mention)
        scored: list[Candidate] = []
        entries = self.store.entries_for_type(entity_type) if entity_type is not None else self.store.entries
        for entry in entries:
            best_score = 0.0
            best_alias: str | None = None
            for alias in entry.all_names:
                alias_norm = normalize_for_match(alias, strip_diacritics=True)
                ratio = SequenceMatcher(a=mention_norm, b=alias_norm).ratio()
                alias_tokens = token_set(alias)
                union = mention_tokens | alias_tokens
                jaccard = len(mention_tokens & alias_tokens) / len(union) if union else 0.0
                score = max(ratio, jaccard)
                if score > best_score:
                    best_score = score
                    best_alias = alias
            if best_score >= self.min_score:
                scored.append(
                    Candidate(
                        concept_id=entry.concept_id,
                        code=entry.code,
                        code_system=entry.code_system,
                        canonical_name=entry.canonical_name,
                        semantic_type=entry.semantic_type,
                        score=best_score,
                        source="fuzzy",
                        matched_alias=best_alias,
                    )
                )
        return sorted(scored, key=lambda candidate: candidate.score, reverse=True)[:limit]

