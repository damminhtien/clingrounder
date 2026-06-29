from __future__ import annotations
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.linking.candidate import Candidate


class ExactMatcher:
    def __init__(self, store: DictionaryStore) -> None:
        self.store = store

    def retrieve(self, mention: str) -> list[Candidate]:
        candidates: list[Candidate] = []
        seen: set[str] = set()
        for entry in [*self.store.exact_lookup(mention), *self.store.toneless_lookup(mention)]:
            if entry.concept_id in seen:
                continue
            seen.add(entry.concept_id)
            candidates.append(
                Candidate(
                    concept_id=entry.concept_id,
                    code=entry.code,
                    code_system=entry.code_system,
                    canonical_name=entry.canonical_name,
                    semantic_type=entry.semantic_type,
                    score=1.0,
                    source="exact",
                    matched_alias=mention,
                )
            )
        return candidates

