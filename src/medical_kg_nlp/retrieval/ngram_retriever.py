from __future__ import annotations

import math
from collections import Counter, defaultdict

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match


class CharNgramRetriever:
    def __init__(
        self,
        store: DictionaryStore,
        min_n: int = 3,
        max_n: int = 5,
        min_score: float = 0.12,
    ) -> None:
        self.store = store
        self.min_n = min_n
        self.max_n = max_n
        self.min_score = min_score
        self.docs: list[Counter[str]] = []
        self.df: Counter[str] = Counter()
        self.postings: dict[str, set[int]] = defaultdict(set)
        for entry in store.entries:
            ngrams: Counter[str] = Counter()
            for alias in entry.all_names:
                ngrams.update(self._ngrams(alias))
            self.docs.append(ngrams)
            self.df.update(set(ngrams))
        for index, ngrams in enumerate(self.docs):
            for ngram in ngrams:
                self.postings[ngram].add(index)
        self.doc_weights = [self._tfidf(doc) for doc in self.docs]
        self.doc_norms = [self._norm(weights) for weights in self.doc_weights]

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType | None = None,
        limit: int = 20,
    ) -> list[Candidate]:
        query = Counter(self._ngrams(mention))
        if not query:
            return []
        query_weights = self._tfidf(query)
        query_norm = self._norm(query_weights)
        if query_norm == 0.0:
            return []

        candidate_indices = set().union(*(self.postings.get(ngram, set()) for ngram in query))
        scored: list[tuple[float, int]] = []
        for index in candidate_indices:
            entry = self.store.entries[index]
            if entity_type is not None and entry.semantic_type != entity_type:
                continue
            doc_weights = self.doc_weights[index]
            doc_norm = self.doc_norms[index]
            if doc_norm == 0.0:
                continue
            score = self._dot(query_weights, doc_weights) / (query_norm * doc_norm)
            if score >= self.min_score:
                scored.append((score, index))

        candidates: list[Candidate] = []
        for score, index in sorted(scored, reverse=True)[:limit]:
            entry = self.store.entries[index]
            candidates.append(
                Candidate(
                    concept_id=entry.concept_id,
                    code=entry.code,
                    code_system=entry.code_system,
                    canonical_name=entry.canonical_name,
                    semantic_type=entry.semantic_type,
                    score=min(1.0, score),
                    source="char_ngram",
                    matched_alias=None,
                )
            )
        return candidates

    def _ngrams(self, text: str) -> list[str]:
        normalized = f" {normalize_for_match(text, strip_diacritics=True)} "
        compact = " ".join(normalized.split())
        grams: list[str] = []
        for size in range(self.min_n, self.max_n + 1):
            if len(compact) < size:
                continue
            grams.extend(compact[index : index + size] for index in range(len(compact) - size + 1))
        return grams

    def _tfidf(self, counts: Counter[str]) -> dict[str, float]:
        total_docs = max(len(self.docs), 1)
        return {
            term: count * math.log(1 + total_docs / (1 + self.df[term]))
            for term, count in counts.items()
        }

    @staticmethod
    def _dot(left: dict[str, float], right: dict[str, float]) -> float:
        return sum(weight * right.get(term, 0.0) for term, weight in left.items())

    @staticmethod
    def _norm(weights: dict[str, float]) -> float:
        return math.sqrt(sum(weight * weight for weight in weights.values()))
