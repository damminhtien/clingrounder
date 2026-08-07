"""Candidate retrieval ports, adapters, and deterministic fusion."""

from __future__ import annotations

from clingrounder.retrieval.dense_retriever import (
    DenseHit,
    DenseRetrieverAdapter,
    DenseVectorIndexPort,
    TextEncoderPort,
)
from clingrounder.retrieval.ngram_retriever import CharNgramRetriever
from clingrounder.retrieval.adapters import KnowledgeGraphExactRetrieverAdapter
from clingrounder.retrieval.pipeline import RetrievalPipeline
from clingrounder.retrieval.query_expansion import (
    RetrievalQueryVariant,
    build_retrieval_query_variants,
)
from clingrounder.retrieval.rule_factory import (
    build_in_memory_retrieval_pipeline,
    build_rule_retrieval_pipeline,
)
from clingrounder.retrieval.synonym_index import (
    FaissSynonymVectorIndex,
    InMemorySynonymVectorIndex,
    SynonymIndexMetadata,
    SynonymVectorRecord,
    build_synonym_vector_records,
    fingerprint_terminology_entries,
    write_faiss_synonym_index,
)

__all__ = [
    "CharNgramRetriever",
    "KnowledgeGraphExactRetrieverAdapter",
    "DenseHit",
    "DenseRetrieverAdapter",
    "DenseVectorIndexPort",
    "FaissSynonymVectorIndex",
    "InMemorySynonymVectorIndex",
    "RetrievalPipeline",
    "RetrievalQueryVariant",
    "TextEncoderPort",
    "SynonymIndexMetadata",
    "SynonymVectorRecord",
    "build_in_memory_retrieval_pipeline",
    "build_retrieval_query_variants",
    "build_rule_retrieval_pipeline",
    "build_synonym_vector_records",
    "fingerprint_terminology_entries",
    "write_faiss_synonym_index",
]
