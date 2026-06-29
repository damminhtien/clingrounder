from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.retrieval.bm25_retriever import BM25Retriever
from medical_kg_nlp.retrieval.exact_matcher import ExactMatcher
from medical_kg_nlp.retrieval.fuzzy_matcher import FuzzyMatcher
from medical_kg_nlp.retrieval.ngram_retriever import CharNgramRetriever
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match


DEFAULT_RETRIEVAL_SOURCES = frozenset({"exact", "abbreviation", "fuzzy", "char_ngram", "bm25"})

ALLOWED_CODE_SYSTEMS: dict[EntityType, set[CodeSystem]] = {
    EntityType.DRUG: {CodeSystem.RXNORM},
    EntityType.DISEASE: {CodeSystem.ICD10, CodeSystem.UMLS, CodeSystem.SNOMED},
    EntityType.SYMPTOM: {CodeSystem.UMLS, CodeSystem.SNOMED, CodeSystem.LOCAL},
    EntityType.LAB_TEST: {CodeSystem.LOCAL},
    EntityType.LAB_RESULT: {CodeSystem.NONE, CodeSystem.LOCAL},
    EntityType.PROCEDURE: {CodeSystem.ICD10, CodeSystem.UMLS, CodeSystem.SNOMED, CodeSystem.LOCAL},
    EntityType.FINDING: {CodeSystem.UMLS, CodeSystem.SNOMED, CodeSystem.LOCAL},
}


class CandidateGenerator:
    def __init__(
        self,
        store: DictionaryStore,
        abbreviation_path: str | Path | None = None,
        max_candidates: int = 20,
        retrieval_sources: tuple[str, ...] | None = None,
    ) -> None:
        self.store = store
        self.max_candidates = max_candidates
        self.retrieval_sources = set(DEFAULT_RETRIEVAL_SOURCES if retrieval_sources is None else retrieval_sources)
        unknown_sources = self.retrieval_sources - DEFAULT_RETRIEVAL_SOURCES
        if unknown_sources:
            raise ValueError(f"Unknown retrieval source(s): {sorted(unknown_sources)}")
        self.exact = ExactMatcher(store)
        self.fuzzy = FuzzyMatcher(store)
        self.char_ngram = CharNgramRetriever(store)
        self.bm25 = BM25Retriever(store)
        self.abbreviations = self._load_abbreviations(abbreviation_path)

    def generate(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str = "",
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        if "exact" in self.retrieval_sources:
            candidates.extend(self.exact.retrieve(mention))
        normalized = normalize_for_match(mention)
        if "abbreviation" in self.retrieval_sources:
            for expansion in self.abbreviations.get(normalized, []):
                for candidate in self.exact.retrieve(expansion):
                    candidates.append(
                        self._replace(candidate, score=0.9, source="abbreviation", matched_alias=expansion)
                    )
        if "fuzzy" in self.retrieval_sources:
            candidates.extend(self.fuzzy.retrieve(mention, entity_type=entity_type, limit=self.max_candidates))
        if "char_ngram" in self.retrieval_sources:
            candidates.extend(self.char_ngram.retrieve(mention, entity_type=entity_type, limit=self.max_candidates))
        if "bm25" in self.retrieval_sources:
            candidates.extend(self.bm25.retrieve(mention, entity_type=entity_type, limit=self.max_candidates))
        constrained = [candidate for candidate in candidates if self._allowed(candidate, entity_type)]
        merged = self._merge(constrained)
        return sorted(merged, key=lambda candidate: candidate.score, reverse=True)[: self.max_candidates]

    def _allowed(self, candidate: Candidate, entity_type: EntityType) -> bool:
        allowed = ALLOWED_CODE_SYSTEMS.get(entity_type)
        if not allowed:
            return candidate.semantic_type == entity_type
        return candidate.semantic_type == entity_type and candidate.code_system in allowed

    def _merge(self, candidates: list[Candidate]) -> list[Candidate]:
        best: dict[str, Candidate] = {}
        source_bonus: dict[str, float] = defaultdict(float)
        for candidate in candidates:
            source_bonus[candidate.concept_id] += 0.03
            current = best.get(candidate.concept_id)
            adjusted = min(1.0, candidate.score + source_bonus[candidate.concept_id])
            improved = self._replace(candidate, score=adjusted)
            if current is None or improved.score > current.score:
                best[candidate.concept_id] = improved
        return list(best.values())

    def _load_abbreviations(self, path: str | Path | None) -> dict[str, list[str]]:
        if path is None or not Path(path).exists():
            return {}
        table: dict[str, list[str]] = defaultdict(list)
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                abbreviation = normalize_for_match(str(row["abbreviation"]))
                expansions = [str(value) for value in row.get("expansions", [])]
                table[abbreviation].extend(expansions)
        return table

    @staticmethod
    def _replace(
        candidate: Candidate,
        *,
        score: float | None = None,
        source: str | None = None,
        matched_alias: str | None = None,
    ) -> Candidate:
        return Candidate(
            concept_id=candidate.concept_id,
            code=candidate.code,
            code_system=candidate.code_system,
            canonical_name=candidate.canonical_name,
            semantic_type=candidate.semantic_type,
            score=candidate.score if score is None else score,
            source=candidate.source if source is None else source,
            matched_alias=candidate.matched_alias if matched_alias is None else matched_alias,
        )
