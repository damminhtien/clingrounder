from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.linking.candidate import Candidate, CandidateEvidence
from medical_kg_nlp.retrieval.bm25_retriever import BM25Retriever
from medical_kg_nlp.retrieval.exact_matcher import ExactMatcher
from medical_kg_nlp.retrieval.fuzzy_matcher import FuzzyMatcher
from medical_kg_nlp.retrieval.ngram_retriever import CharNgramRetriever
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match


DEFAULT_RETRIEVAL_SOURCES = frozenset({"exact", "abbreviation", "fuzzy", "char_ngram", "bm25"})
SOURCE_WEIGHTS = {
    "exact": 1.0,
    "toneless": 0.92,
    "abbreviation": 0.92,
    "fuzzy": 0.72,
    "char_ngram": 0.62,
    "bm25": 0.50,
}
_RRF_K = 60
_PRIMARY_SCORE_WEIGHT = 0.85

ALLOWED_CODE_SYSTEMS: dict[EntityType, set[CodeSystem]] = {
    EntityType.DRUG: {CodeSystem.RXNORM},
    EntityType.DISEASE: {CodeSystem.ICD10, CodeSystem.UMLS, CodeSystem.SNOMED},
    EntityType.SYMPTOM: {CodeSystem.UMLS, CodeSystem.SNOMED, CodeSystem.LOCAL},
    EntityType.LAB_TEST: {CodeSystem.LOCAL},
    EntityType.LAB_RESULT: {CodeSystem.NONE, CodeSystem.LOCAL},
    EntityType.DOSAGE: {CodeSystem.NONE},
    EntityType.STRENGTH: {CodeSystem.NONE},
    EntityType.FREQUENCY: {CodeSystem.NONE},
    EntityType.ROUTE: {CodeSystem.NONE},
    EntityType.DURATION: {CodeSystem.NONE},
    EntityType.DOSAGE_FORM: {CodeSystem.NONE},
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
        self.retrieval_sources = set(
            DEFAULT_RETRIEVAL_SOURCES if retrieval_sources is None else retrieval_sources
        )
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
            exact_candidates = self.exact.retrieve(mention)
            candidates.extend(exact_candidates)
            unique_exact = self._unique_exact_output(exact_candidates, entity_type)
            if unique_exact is not None:
                return [unique_exact]
        normalized = normalize_for_match(mention)
        if "abbreviation" in self.retrieval_sources:
            for expansion in self.abbreviations.get(normalized, []):
                for candidate in self.exact.retrieve(expansion):
                    candidates.append(
                        self._replace(
                            candidate, score=0.9, source="abbreviation", matched_alias=expansion
                        )
                    )
        if "fuzzy" in self.retrieval_sources:
            candidates.extend(
                self.fuzzy.retrieve(mention, entity_type=entity_type, limit=self.max_candidates)
            )
        if "char_ngram" in self.retrieval_sources:
            candidates.extend(
                self.char_ngram.retrieve(
                    mention, entity_type=entity_type, limit=self.max_candidates
                )
            )
        if "bm25" in self.retrieval_sources:
            candidates.extend(
                self.bm25.retrieve(mention, entity_type=entity_type, limit=self.max_candidates)
            )
        constrained = [
            candidate for candidate in candidates if self._allowed(candidate, entity_type)
        ]
        merged = self._merge(constrained)
        return sorted(merged, key=lambda candidate: candidate.score, reverse=True)[
            : self.max_candidates
        ]

    def _unique_exact_output(
        self,
        candidates: list[Candidate],
        entity_type: EntityType,
    ) -> Candidate | None:
        exact = [
            candidate
            for candidate in candidates
            if candidate.source == "exact" and self._allowed(candidate, entity_type)
        ]
        if not exact or any(candidate.code is None for candidate in exact):
            return None
        output_keys = {self._output_key(candidate) for candidate in exact}
        if len(output_keys) != 1:
            return None
        merged = self._merge(exact)
        if len(merged) != 1:
            return None
        return self._replace(merged[0], score=max(candidate.score for candidate in exact))

    def _allowed(self, candidate: Candidate, entity_type: EntityType) -> bool:
        allowed = ALLOWED_CODE_SYSTEMS.get(entity_type)
        if not allowed:
            return candidate.semantic_type == entity_type
        return candidate.semantic_type == entity_type and candidate.code_system in allowed

    def _merge(self, candidates: list[Candidate]) -> list[Candidate]:
        by_source: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            by_source[candidate.source].append(candidate)

        evidence_by_code: dict[tuple[str, str], dict[str, tuple[CandidateEvidence, Candidate]]] = (
            defaultdict(dict)
        )
        for source, source_candidates in sorted(by_source.items()):
            ranked = sorted(
                source_candidates,
                key=lambda item: (
                    -item.score,
                    item.code_system.value,
                    item.code or "",
                    item.concept_id,
                ),
            )
            for rank, candidate in enumerate(ranked, start=1):
                key = self._output_key(candidate)
                evidence = CandidateEvidence(
                    source=source,
                    score=candidate.score,
                    rank=rank,
                    concept_id=candidate.concept_id,
                    matched_alias=candidate.matched_alias,
                )
                current = evidence_by_code[key].get(source)
                if current is None or self._evidence_order(evidence) > self._evidence_order(
                    current[0]
                ):
                    evidence_by_code[key][source] = (evidence, candidate)

        merged: list[Candidate] = []
        rrf_normalizer = sum(weight / (_RRF_K + 1) for weight in SOURCE_WEIGHTS.values())
        for source_evidence in evidence_by_code.values():
            ordered = sorted(
                source_evidence.values(),
                key=lambda item: (
                    -(SOURCE_WEIGHTS.get(item[0].source, 0.0) * item[0].score),
                    item[0].rank,
                    item[0].source,
                    item[0].concept_id,
                ),
            )
            primary_evidence, primary = ordered[0]
            merged_evidence = tuple(item[0] for item in ordered)
            primary_score = max(
                SOURCE_WEIGHTS.get(item.source, 0.0) * item.score for item in merged_evidence
            )
            rrf_score = sum(
                SOURCE_WEIGHTS.get(item.source, 0.0) * item.score / (_RRF_K + item.rank)
                for item in merged_evidence
            )
            normalized_rrf = rrf_score / rrf_normalizer if rrf_normalizer else 0.0
            fused_score = min(
                1.0,
                _PRIMARY_SCORE_WEIGHT * primary_score
                + (1.0 - _PRIMARY_SCORE_WEIGHT) * normalized_rrf,
            )
            merged.append(
                self._replace(
                    primary,
                    score=fused_score,
                    source=primary_evidence.source,
                    matched_alias=primary_evidence.matched_alias,
                    evidence=merged_evidence,
                )
            )
        return merged

    @staticmethod
    def _output_key(candidate: Candidate) -> tuple[str, str]:
        if candidate.code:
            return candidate.code_system.value, candidate.code
        return "concept", candidate.concept_id

    @staticmethod
    def _evidence_order(evidence: CandidateEvidence) -> tuple[float, int, str]:
        return evidence.score, -evidence.rank, evidence.concept_id

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
        evidence: tuple[CandidateEvidence, ...] | None = None,
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
            evidence=candidate.evidence if evidence is None else evidence,
        )
