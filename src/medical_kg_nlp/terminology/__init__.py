"""Terminology repositories and persistent derived-index tooling."""

from medical_kg_nlp.terminology.composite import CompositeTerminologyRepository
from medical_kg_nlp.terminology.evaluation import (
    TerminologyQuery,
    evaluate_terminology_queries,
    load_terminology_queries,
)
from medical_kg_nlp.terminology.index_builder import (
    TERMINOLOGY_INDEX_SCHEMA_VERSION,
    TerminologyIndexManifest,
    build_terminology_index,
    input_fingerprint,
    source_fingerprint,
    terminology_cache_path,
)
from medical_kg_nlp.terminology.memory import InMemoryTerminologyRepository
from medical_kg_nlp.terminology.ports import TerminologyRepository
from medical_kg_nlp.terminology.sqlite_repository import SQLiteTerminologyRepository

__all__ = [
    "CompositeTerminologyRepository",
    "InMemoryTerminologyRepository",
    "SQLiteTerminologyRepository",
    "TERMINOLOGY_INDEX_SCHEMA_VERSION",
    "TerminologyRepository",
    "TerminologyQuery",
    "TerminologyIndexManifest",
    "build_terminology_index",
    "evaluate_terminology_queries",
    "input_fingerprint",
    "load_terminology_queries",
    "source_fingerprint",
    "terminology_cache_path",
]
