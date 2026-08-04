"""Terminology repositories and persistent derived-index tooling."""

from medical_kg_nlp.terminology.cache import (
    CachedTerminologyRepository,
    TerminologyCacheInfo,
)
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
from medical_kg_nlp.terminology.ports import (
    TerminologyMembershipPort,
    TerminologyRepository,
    TerminologySearchHit,
)
from medical_kg_nlp.terminology.query_sets import (
    build_alias_overlay_queries,
    build_linked_proposal_queries,
    write_alias_overlay_query_set,
    write_linked_proposal_query_set,
)
from medical_kg_nlp.terminology.sqlite_repository import SQLiteTerminologyRepository

__all__ = [
    "CachedTerminologyRepository",
    "CompositeTerminologyRepository",
    "InMemoryTerminologyRepository",
    "SQLiteTerminologyRepository",
    "TERMINOLOGY_INDEX_SCHEMA_VERSION",
    "TerminologyMembershipPort",
    "TerminologyRepository",
    "TerminologySearchHit",
    "TerminologyCacheInfo",
    "TerminologyQuery",
    "TerminologyIndexManifest",
    "build_alias_overlay_queries",
    "build_linked_proposal_queries",
    "build_terminology_index",
    "evaluate_terminology_queries",
    "input_fingerprint",
    "load_terminology_queries",
    "source_fingerprint",
    "terminology_cache_path",
    "write_alias_overlay_query_set",
    "write_linked_proposal_query_set",
]
