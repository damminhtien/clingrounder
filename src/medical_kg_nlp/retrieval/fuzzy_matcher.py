from __future__ import annotations
from collections import defaultdict
from difflib import SequenceMatcher

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match, token_set


class FuzzyMatcher:
    def __init__(self, store: DictionaryStore, min_score: float = 0.72) -> None:
        self.store = store
        self.min_score = min_score
        self.postings: dict[str, set[int]] = defaultdict(set)
        for index, entry in enumerate(store.entries):
            for alias in entry.all_names:
                for ngram in self._trigrams(alias):
                    self.postings[ngram].add(index)

    def retrieve(
        self, mention: str, entity_type: EntityType | None = None, limit: int = 20
    ) -> list[Candidate]:
        mention_norm = normalize_for_match(mention, strip_diacritics=True)
        mention_tokens = token_set(mention)
        query_ngrams = self._trigrams(mention)
        if not query_ngrams:
            return []
        scored: list[Candidate] = []
        candidate_indices = set().union(
            *(self.postings.get(ngram, set()) for ngram in query_ngrams)
        )
        for index in candidate_indices:
            entry = self.store.entries[index]
            if entity_type is not None and entry.semantic_type != entity_type:
                continue
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

    @staticmethod
    def _trigrams(text: str) -> frozenset[str]:
        normalized = f" {normalize_for_match(text, strip_diacritics=True)} "
        compact = " ".join(normalized.split())
        if len(compact) < 3:
            return frozenset()
        return frozenset(compact[index : index + 3] for index in range(len(compact) - 2))
