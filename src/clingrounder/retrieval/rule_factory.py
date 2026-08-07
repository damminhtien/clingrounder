"""Build deterministic retriever adapters without coupling RetrievalPipeline to storage."""

from __future__ import annotations

from pathlib import Path

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.linking.learned_edits import (
    LearnedEditModel,
    LearnedEditRetrieverAdapter,
)
from clingrounder.linking.mention_code_memory import (
    MentionCodeMemory,
    MentionCodeMemoryRetrieverAdapter,
)
from clingrounder.retrieval.adapters import (
    AbbreviationRetrieverAdapter,
    BM25RetrieverAdapter,
    CharNgramRetrieverAdapter,
    ExactRetrieverAdapter,
    FTSRetrieverAdapter,
    FuzzyRetrieverAdapter,
    KnowledgeGraphExactRetrieverAdapter,
    MentionRetrieverAdapter,
    ReviewedMentionRetrieverAdapter,
)
from clingrounder.retrieval.bm25_retriever import BM25Retriever
from clingrounder.retrieval.fuzzy_matcher import FuzzyMatcher
from clingrounder.retrieval.ngram_retriever import CharNgramRetriever
from clingrounder.retrieval.pipeline import RetrievalPipeline
from clingrounder.retrieval.dense_retriever import DenseRetrieverAdapter
from clingrounder.terminology.memory import InMemoryTerminologyRepository
from clingrounder.terminology.ports import TerminologyRepository
from clingrounder.kg.ports import KnowledgeGraphRepositoryPort

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
    mention_memory_path: str | Path | None = None,
    mention_code_memory: MentionCodeMemory | None = None,
    learned_edit_model: LearnedEditModel | None = None,
    dense_retriever: DenseRetrieverAdapter | None = None,
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
        mention_code_memory=mention_code_memory,
        learned_edit_model=learned_edit_model,
        dense_retriever=dense_retriever,
    )


def build_rule_retrieval_pipeline(
    repository: TerminologyRepository,
    *,
    approximate_store: DictionaryStore,
    abbreviation_path: str | Path | None = None,
    max_candidates: int = 20,
    retrieval_sources: tuple[str, ...] | None = None,
    mention_memory_path: str | Path | None = None,
    use_fts_for_bm25: bool = False,
    knowledge_graph_repository: KnowledgeGraphRepositoryPort | None = None,
    mention_code_memory: MentionCodeMemory | None = None,
    learned_edit_model: LearnedEditModel | None = None,
    dense_retriever: DenseRetrieverAdapter | None = None,
) -> RetrievalPipeline:
    """Compose selected retrievers while keeping the pipeline storage-neutral."""

    selected = set(retrieval_sources or DEFAULT_RETRIEVAL_SOURCES)
    supported = {
        *DEFAULT_RETRIEVAL_SOURCES,
        "dense",
        "kg_exact",
        "learned_edit",
        "mention_memory",
    }
    unknown = selected - supported
    if unknown:
        raise ValueError(f"Unknown retrieval source(s): {sorted(unknown)}")

    adapters: list[MentionRetrieverAdapter] = [
        ReviewedMentionRetrieverAdapter.from_jsonl(repository, mention_memory_path)
    ]
    if "mention_memory" in selected:
        if mention_code_memory is None:
            raise ValueError("mention_memory retrieval requires a mention-code memory artifact")
        adapters.append(
            MentionCodeMemoryRetrieverAdapter(
                mention_code_memory,
                repository,
                high_confidence_only=True,
            )
        )
    if "exact" in selected:
        adapters.append(ExactRetrieverAdapter(repository))
    if "abbreviation" in selected:
        adapters.append(AbbreviationRetrieverAdapter.from_jsonl(repository, abbreviation_path))
    if "learned_edit" in selected:
        if learned_edit_model is None:
            raise ValueError("learned_edit retrieval requires a learned-edit artifact")
        adapters.append(LearnedEditRetrieverAdapter(learned_edit_model, repository))
    if "mention_memory" in selected:
        if mention_code_memory is None:  # pragma: no cover - guarded above
            raise RuntimeError("Mention-code memory composition invariant failed")
        adapters.append(
            MentionCodeMemoryRetrieverAdapter(
                mention_code_memory,
                repository,
                high_confidence_only=False,
            )
        )
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
    if "dense" in selected:
        if dense_retriever is None:
            raise ValueError("dense retrieval requires an encoder and vector index")
        adapters.append(dense_retriever)
    if "kg_exact" in selected:
        if knowledge_graph_repository is None:
            raise ValueError("kg_exact retrieval requires a knowledge graph repository")
        adapters.append(KnowledgeGraphExactRetrieverAdapter(knowledge_graph_repository))
    return RetrievalPipeline(
        repository,
        tuple(adapters),
        max_candidates=max_candidates,
    )
