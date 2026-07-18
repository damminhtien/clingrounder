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
from medical_kg_nlp.retrieval.rule_factory import (
    build_in_memory_retrieval_pipeline,
    build_rule_retrieval_pipeline,
)

__all__ = [
    "CharNgramRetriever",
    "KnowledgeGraphExactRetrieverAdapter",
    "DenseHit",
    "DenseRetrieverAdapter",
    "DenseVectorIndexPort",
    "RetrievalPipeline",
    "TextEncoderPort",
    "build_in_memory_retrieval_pipeline",
    "build_rule_retrieval_pipeline",
]
