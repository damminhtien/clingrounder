"""Build deterministic retriever adapters without coupling RetrievalPipeline to storage."""

from __future__ import annotations

from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.retrieval.adapters import (
    AbbreviationRetrieverAdapter,
    BM25RetrieverAdapter,
    CharNgramRetrieverAdapter,
    ExactRetrieverAdapter,
    FTSRetrieverAdapter,
    FuzzyRetrieverAdapter,
    MentionRetrieverAdapter,
    ReviewedMentionRetrieverAdapter,
)
from medical_kg_nlp.retrieval.bm25_retriever import BM25Retriever
from medical_kg_nlp.retrieval.fuzzy_matcher import FuzzyMatcher
from medical_kg_nlp.retrieval.ngram_retriever import CharNgramRetriever
from medical_kg_nlp.retrieval.pipeline import RetrievalPipeline
from medical_kg_nlp.terminology.memory import InMemoryTerminologyRepository
from medical_kg_nlp.terminology.ports import TerminologyRepository

__all__ = [
    "DEFAULT_RETRIEVAL_SOURCES",
    "build_in_memory_retrieval_pipeline",
    "build_rule_retrieval_pipeline",
]

DEFAULT_RETRIEVAL_SOURCES = ("exact", "abbreviation", "fuzzy", "char_ngram", "bm25")


def build_in_memory_retrieval_pipeline(
    store: DictionaryStore,
    abbreviation_path: str | Path | None = None,
    max_candidates: int = 20,
    retrieval_sources: tuple[str, ...] | None = None,
    mention_memory_path: str | Path | None = (
        "src/medical_kg_nlp/resources/phase1_rxnorm_memory.jsonl"
    ),
) -> RetrievalPipeline:
    """Compose the default rule retrievers over an in-memory terminology."""

    repository = InMemoryTerminologyRepository(store)
    return build_rule_retrieval_pipeline(
        repository,
        approximate_store=store,
        abbreviation_path=abbreviation_path,
        max_candidates=max_candidates,
        retrieval_sources=retrieval_sources,
        mention_memory_path=mention_memory_path,
    )


def build_rule_retrieval_pipeline(
    repository: TerminologyRepository,
    *,
    approximate_store: DictionaryStore,
    abbreviation_path: str | Path | None = None,
    max_candidates: int = 20,
    retrieval_sources: tuple[str, ...] | None = None,
    mention_memory_path: str | Path | None = (
        "src/medical_kg_nlp/resources/phase1_rxnorm_memory.jsonl"
    ),
    use_fts_for_bm25: bool = False,
) -> RetrievalPipeline:
    """Compose selected retrievers while keeping the pipeline storage-neutral."""

    selected = set(retrieval_sources or DEFAULT_RETRIEVAL_SOURCES)
    unknown = selected - set(DEFAULT_RETRIEVAL_SOURCES)
    if unknown:
        raise ValueError(f"Unknown retrieval source(s): {sorted(unknown)}")

    adapters: list[MentionRetrieverAdapter] = [
        ReviewedMentionRetrieverAdapter.from_jsonl(repository, mention_memory_path)
    ]
    if "exact" in selected:
        adapters.append(ExactRetrieverAdapter(repository))
    if "abbreviation" in selected:
        adapters.append(AbbreviationRetrieverAdapter.from_jsonl(repository, abbreviation_path))
    # SCALING: approximate in-memory indexes stay recognition-sized when normalization uses
    # SQLite. FTS is the persistent lexical source over the complete terminology.
    if "fuzzy" in selected:
        adapters.append(FuzzyRetrieverAdapter(FuzzyMatcher(approximate_store)))
    if "char_ngram" in selected:
        adapters.append(CharNgramRetrieverAdapter(CharNgramRetriever(approximate_store)))
    if "bm25" in selected:
        adapters.append(
            FTSRetrieverAdapter(repository)
            if use_fts_for_bm25
            else BM25RetrieverAdapter(BM25Retriever(approximate_store))
        )
    return RetrievalPipeline(
        repository,
        tuple(adapters),
        max_candidates=max_candidates,
    )
