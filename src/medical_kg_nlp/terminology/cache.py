"""Bounded, thread-safe cache for immutable terminology repositories."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["CachedTerminologyRepository", "TerminologyCacheInfo"]

_T = TypeVar("_T")
_MISSING = object()


@dataclass(frozen=True)
class TerminologyCacheInfo:
    """Observable cache state used by benchmarks and operational traces."""

    hits: int
    misses: int
    max_size: int
    current_size: int


class CachedTerminologyRepository:
    """Cache deterministic repository calls without changing retrieval semantics.

    The wrapped repository and its source fingerprint remain the source of truth. Values
    are immutable concept records or tuples, and list-returning APIs always return a copy.
    """

    def __init__(self, repository: TerminologyRepository, *, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("Terminology query cache size must be positive")
        self.repository = repository
        self.max_size = max_size
        self._entries: OrderedDict[tuple[object, ...], object] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get_by_concept_id(self, concept_id: str) -> ConceptEntry | None:
        return self._get_or_load(
            ("concept", concept_id),
            lambda: self.repository.get_by_concept_id(concept_id),
        )

    def get_by_code(self, code_system: CodeSystem, code: str) -> ConceptEntry | None:
        return self._get_or_load(
            ("code", code_system.value, code),
            lambda: self.repository.get_by_code(code_system, code),
        )

    def exact_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._cached_query(
            "exact",
            normalize_for_match(mention),
            entity_type,
            code_systems,
            limit,
            lambda: self.repository.exact_lookup(
                mention,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=limit,
            ),
        )

    def toneless_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._cached_query(
            "toneless",
            normalize_for_match(mention, strip_diacritics=True),
            entity_type,
            code_systems,
            limit,
            lambda: self.repository.toneless_lookup(
                mention,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=limit,
            ),
        )

    def search(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._cached_query(
            "search",
            normalize_for_match(mention),
            entity_type,
            code_systems,
            limit,
            lambda: self.repository.search(
                mention,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=limit,
            ),
        )

    def cache_info(self) -> TerminologyCacheInfo:
        """Return a consistent snapshot without exposing mutable cache entries."""

        with self._lock:
            return TerminologyCacheInfo(
                hits=self._hits,
                misses=self._misses,
                max_size=self.max_size,
                current_size=len(self._entries),
            )

    def clear(self) -> None:
        """Clear values and counters, primarily for isolated benchmarks."""

        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def _cached_query(
        self,
        operation: str,
        normalized_mention: str,
        entity_type: EntityType | None,
        code_systems: Sequence[CodeSystem] | None,
        limit: int,
        loader: Callable[[], list[ConceptEntry]],
    ) -> list[ConceptEntry]:
        key = (
            operation,
            normalized_mention,
            None if entity_type is None else entity_type.value,
            _code_system_key(code_systems),
            limit,
        )
        values = self._get_or_load(key, lambda: tuple(loader()))
        # INVARIANT: callers may reorder or trim their result without mutating the cached value.
        return list(values)

    def _get_or_load(self, key: tuple[object, ...], loader: Callable[[], _T]) -> _T:
        with self._lock:
            cached = self._entries.pop(key, _MISSING)
            if cached is not _MISSING:
                self._entries[key] = cached
                self._hits += 1
                return cast(_T, cached)

        # SCALING: do not serialize independent SQLite misses behind the cache lock. Two workers
        # may compute the same cold key, but publication remains deterministic and bounded.
        loaded = loader()
        with self._lock:
            existing = self._entries.pop(key, _MISSING)
            if existing is not _MISSING:
                self._entries[key] = existing
                self._hits += 1
                return cast(_T, existing)
            self._entries[key] = loaded
            self._misses += 1
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
        return loaded


def _code_system_key(
    code_systems: Sequence[CodeSystem] | None,
) -> tuple[str, ...] | None:
    if code_systems is None:
        return None
    return tuple(sorted({code_system.value for code_system in code_systems}))
