"""Bounded terminology cache contract tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from clingrounder.dictionaries.synonym_table import ConceptEntry
from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.terminology import CachedTerminologyRepository


def _concept() -> ConceptEntry:
    return ConceptEntry(
        concept_id="ICD:I10",
        code="I10",
        code_system=CodeSystem.ICD10,
        canonical_name="tăng huyết áp",
        semantic_type=EntityType.DISEASE,
        source="fixture",
    )


@dataclass
class _CountingRepository:
    entry: ConceptEntry
    calls: dict[str, int] = field(default_factory=dict)

    def _record(self, operation: str) -> None:
        self.calls[operation] = self.calls.get(operation, 0) + 1

    def get_by_concept_id(self, concept_id: str) -> ConceptEntry | None:
        self._record("concept")
        return self.entry if concept_id == self.entry.concept_id else None

    def get_by_code(self, code_system: CodeSystem, code: str) -> ConceptEntry | None:
        self._record("code")
        return self.entry if (code_system, code) == (self.entry.code_system, self.entry.code) else None

    def contains(self, code_system: CodeSystem, code: str) -> bool:
        self._record("contains")
        return (code_system, code) == (self.entry.code_system, self.entry.code)

    def exact_lookup(self, mention: str, **_: object) -> list[ConceptEntry]:
        self._record("exact")
        return [self.entry] if mention.casefold() in {"tăng huyết áp", "TĂNG HUYẾT ÁP".casefold()} else []

    def toneless_lookup(self, mention: str, **_: object) -> list[ConceptEntry]:
        self._record("toneless")
        return [self.entry]

    def search(self, mention: str, **_: object) -> list[ConceptEntry]:
        self._record("search")
        return [self.entry]


def test_cache_reuses_normalized_queries_and_returns_list_copies() -> None:
    source = _CountingRepository(_concept())
    repository = CachedTerminologyRepository(source, max_size=8)

    first = repository.exact_lookup("Tăng huyết áp", entity_type=EntityType.DISEASE)
    first.clear()
    second = repository.exact_lookup("tăng huyết áp", entity_type=EntityType.DISEASE)

    assert [entry.code for entry in second] == ["I10"]
    assert source.calls == {"exact": 1}
    assert repository.cache_info().hits == 1
    assert repository.cache_info().misses == 1


def test_cache_is_bounded_and_caches_missing_identifiers() -> None:
    source = _CountingRepository(_concept())
    repository = CachedTerminologyRepository(source, max_size=2)

    assert repository.get_by_concept_id("missing") is None
    assert repository.get_by_concept_id("missing") is None
    repository.get_by_code(CodeSystem.ICD10, "I10")
    repository.search("hypertension")

    assert source.calls["concept"] == 1
    assert repository.cache_info().current_size == 2


def test_cache_reuses_membership_results() -> None:
    source = _CountingRepository(_concept())
    repository = CachedTerminologyRepository(source, max_size=8)

    assert repository.contains(CodeSystem.ICD10, "I10")
    assert repository.contains(CodeSystem.ICD10, "I10")
    assert not repository.contains(CodeSystem.ICD10, "MISSING")

    assert source.calls == {"contains": 2}


def test_warm_cache_is_deterministic_across_threads() -> None:
    source = _CountingRepository(_concept())
    repository = CachedTerminologyRepository(source, max_size=8)
    repository.search("hypertension", entity_type=EntityType.DISEASE)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: repository.search(
                    "hypertension",
                    entity_type=EntityType.DISEASE,
                ),
                range(64),
            )
        )

    assert [[entry.code for entry in result] for result in results] == [["I10"]] * 64
    assert source.calls == {"search": 1}
    assert repository.cache_info().hits == 64
