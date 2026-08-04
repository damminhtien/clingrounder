"""Candidate retrieval ports, adapters, and deterministic fusion."""

from __future__ import annotations

from medical_kg_nlp.retrieval.dense_retriever import (
    DenseHit,
    DenseRetrieverAdapter,
    DenseVectorIndexPort,
    TextEncoderPort,
)
from medical_kg_nlp.retrieval.ngram_retriever import CharNgramRetriever
from medical_kg_nlp.retrieval.adapters import KnowledgeGraphExactRetrieverAdapter
from medical_kg_nlp.retrieval.pipeline import RetrievalPipeline
from medical_kg_nlp.retrieval.query_expansion import (
    RetrievalQueryVariant,
    build_retrieval_query_variants,
)
from medical_kg_nlp.retrieval.rule_factory import (
    build_in_memory_retrieval_pipeline,
    build_rule_retrieval_pipeline,
)
from medical_kg_nlp.retrieval.synonym_index import (
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
