from __future__ import annotations
import math
from collections import Counter, defaultdict

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.linking.candidate import Candidate
from clingrounder.schema.types import EntityType
from clingrounder.utils.text import normalize_for_match


class BM25Retriever:
    def __init__(self, store: DictionaryStore, k1: float = 1.2, b: float = 0.75) -> None:
        self.store = store
        self.k1 = k1
        self.b = b
        self.docs: list[tuple[int, Counter[str], int]] = []
        self.df: Counter[str] = Counter()
        self.postings: dict[str, set[int]] = defaultdict(set)
        for index, entry in enumerate(store.entries):
            text = " ".join(entry.all_names)
            tokens = normalize_for_match(text, strip_diacritics=True).split()
            counts = Counter(tokens)
            self.docs.append((index, counts, len(tokens)))
            self.df.update(set(tokens))
            for token in counts:
                self.postings[token].add(index)
        self.avgdl = sum(length for _, _, length in self.docs) / max(len(self.docs), 1)

    def retrieve(
        self, mention: str, entity_type: EntityType | None = None, limit: int = 20
    ) -> list[Candidate]:
        query_terms = tuple(
            dict.fromkeys(normalize_for_match(mention, strip_diacritics=True).split())
        )
        if not query_terms:
            return []
        scores: list[tuple[float, int]] = []
        total_docs = max(len(self.docs), 1)
        candidate_indices = set().union(*(self.postings.get(term, set()) for term in query_terms))
        for index in candidate_indices:
            _, counts, doc_len = self.docs[index]
            entry = self.store.entries[index]
            if entity_type is not None and entry.semantic_type != entity_type:
                continue
            score = 0.0
            for term in query_terms:
                if counts[term] == 0:
                    continue
                idf = math.log(1 + (total_docs - self.df[term] + 0.5) / (self.df[term] + 0.5))
                numerator = counts[term] * (self.k1 + 1)
                denominator = counts[term] + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9)
                )
                score += idf * numerator / denominator
            if score > 0:
                scores.append((score, index))
        if not scores:
            return []
        candidates: list[Candidate] = []
        for score, index in sorted(scores, reverse=True)[:limit]:
            entry = self.store.entries[index]
            candidates.append(
                Candidate(
                    concept_id=entry.concept_id,
                    code=entry.code,
                    code_system=entry.code_system,
                    canonical_name=entry.canonical_name,
                    semantic_type=entry.semantic_type,
                    score=self._calibrate(score, len(query_terms)),
                    source="bm25",
                    matched_alias=None,
                )
            )
        return candidates

    @staticmethod
    def _calibrate(raw_score: float, query_term_count: int) -> float:
        """Map BM25 to a fixed scale instead of normalizing by each query's maximum."""

        scale = 4.0 * max(query_term_count, 1)
        return raw_score / (raw_score + scale)
