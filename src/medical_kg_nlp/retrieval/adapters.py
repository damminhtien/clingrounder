"""Mention-level retriever adapters used by the retrieval pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.kg.ports import KnowledgeGraphRepositoryPort
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.retrieval.bm25_retriever import BM25Retriever
from medical_kg_nlp.retrieval.constraints import allowed_code_systems
from medical_kg_nlp.retrieval.fuzzy_matcher import FuzzyMatcher
from medical_kg_nlp.retrieval.ngram_retriever import CharNgramRetriever
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository
from medical_kg_nlp.utils.io import read_jsonl
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "AbbreviationRetrieverAdapter",
    "BM25RetrieverAdapter",
    "CharNgramRetrieverAdapter",
    "ExactRetrieverAdapter",
    "FTSRetrieverAdapter",
    "FuzzyRetrieverAdapter",
    "KnowledgeGraphExactRetrieverAdapter",
    "MentionRetrieverAdapter",
    "ReviewedMentionRetrieverAdapter",
]


class MentionRetrieverAdapter(Protocol):
    """Retrieve candidates for one mention from one evidence source."""

    @property
    def source(self) -> str: ...

    @property
    def terminal_on_match(self) -> bool: ...

    @property
    def unique_output_short_circuit(self) -> bool: ...

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]: ...


@dataclass(frozen=True)
class ExactRetrieverAdapter:
    """Resolve normalized and toneless exact matches from a repository."""

    repository: TerminologyRepository
    source: str = "exact"
    terminal_on_match: bool = False
    unique_output_short_circuit: bool = True

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]:
        del context_window
        systems = allowed_code_systems(entity_type)
        query_limit = max(256, limit)
        exact = self.repository.exact_lookup(
            mention,
            entity_type=entity_type,
            code_systems=systems,
            limit=query_limit,
        )
        output = [_candidate(entry, 1.0, "exact", mention) for entry in exact]
        seen = {entry.concept_id for entry in exact}
        toneless = self.repository.toneless_lookup(
            mention,
            entity_type=entity_type,
            code_systems=systems,
            limit=query_limit,
        )
        output.extend(
            _candidate(entry, 0.92, "toneless", mention)
            for entry in toneless
            if entry.concept_id not in seen
        )
        return output


@dataclass(frozen=True)
class AbbreviationRetrieverAdapter:
    """Expand reviewed abbreviations before exact terminology lookup."""

    exact: ExactRetrieverAdapter
    expansions: dict[str, tuple[str, ...]]
    source: str = "abbreviation"
    terminal_on_match: bool = False
    unique_output_short_circuit: bool = False

    @classmethod
    def from_jsonl(
        cls,
        repository: TerminologyRepository,
        path: str | Path | None,
    ) -> "AbbreviationRetrieverAdapter":
        table: dict[str, list[str]] = defaultdict(list)
        if path is not None and Path(path).exists():
            for row in read_jsonl(path):
                abbreviation = normalize_for_match(str(row["abbreviation"]))
                table[abbreviation].extend(str(value) for value in row.get("expansions", []))
        return cls(
            exact=ExactRetrieverAdapter(repository),
            expansions={key: tuple(values) for key, values in table.items()},
        )

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]:
        output: list[Candidate] = []
        for expansion in self.expansions.get(normalize_for_match(mention), ()):
            output.extend(
                _candidate_from_candidate(candidate, 0.9, self.source, expansion)
                for candidate in self.exact.retrieve(
                    expansion,
                    entity_type,
                    context_window,
                    limit,
                )
            )
        return output


@dataclass(frozen=True)
class ReviewedMentionRetrieverAdapter:
    """Apply human-reviewed mention mappings ahead of approximate retrieval."""

    repository: TerminologyRepository
    memory: dict[str, tuple[CodeSystem, str, str]]
    source: str = "reviewed_memory"
    terminal_on_match: bool = True
    unique_output_short_circuit: bool = False

    @classmethod
    def from_jsonl(
        cls,
        repository: TerminologyRepository,
        path: str | Path | None,
    ) -> "ReviewedMentionRetrieverAdapter":
        memory: dict[str, tuple[CodeSystem, str, str]] = {}
        if path is not None and Path(path).exists():
            for row in read_jsonl(path):
                memory[normalize_for_match(str(row["mention"]))] = (
                    CodeSystem(str(row["code_system"])),
                    str(row["code"]),
                    str(row["provenance"]),
                )
        return cls(repository=repository, memory=memory)

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]:
        del context_window, limit
        remembered = self.memory.get(normalize_for_match(mention))
        if remembered is None:
            return []
        code_system, code, provenance = remembered
        entry = self.repository.get_by_code(code_system, code)
        if entry is None or entry.semantic_type != entity_type:
            return []
        return [
            _candidate(
                entry,
                1.0,
                provenance,
                mention,
                reviewed_mapping=True,
            )
        ]


@dataclass(frozen=True)
class FuzzyRetrieverAdapter:
    """Adapt the bounded in-memory edit-distance matcher to the retrieval port."""

    implementation: FuzzyMatcher
    source: str = "fuzzy"
    terminal_on_match: bool = False
    unique_output_short_circuit: bool = False

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]:
        del context_window
        return self.implementation.retrieve(mention, entity_type=entity_type, limit=limit)


@dataclass(frozen=True)
class CharNgramRetrieverAdapter:
    """Adapt the recognition-sized character n-gram index."""

    implementation: CharNgramRetriever
    source: str = "char_ngram"
    terminal_on_match: bool = False
    unique_output_short_circuit: bool = False

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]:
        del context_window
        return self.implementation.retrieve(mention, entity_type=entity_type, limit=limit)


@dataclass(frozen=True)
class BM25RetrieverAdapter:
    """Adapt the legacy in-memory BM25 retriever for small dictionaries."""

    implementation: BM25Retriever
    source: str = "bm25"
    terminal_on_match: bool = False
    unique_output_short_circuit: bool = False

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]:
        del context_window
        return self.implementation.retrieve(mention, entity_type=entity_type, limit=limit)


@dataclass(frozen=True)
class FTSRetrieverAdapter:
    """Use SQLite FTS5 for lexical retrieval over full terminology sources."""

    repository: TerminologyRepository
    source: str = "bm25"
    terminal_on_match: bool = False
    unique_output_short_circuit: bool = False

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]:
        del context_window
        hits = self.repository.search_scored(
            mention,
            entity_type=entity_type,
            code_systems=allowed_code_systems(entity_type),
            limit=limit,
        )
        return [
            _candidate(
                hit.entry,
                hit.score,
                self.source,
                hit.matched_alias,
            )
            for hit in hits
        ]


@dataclass(frozen=True)
class KnowledgeGraphExactRetrieverAdapter:
    """Retrieve only code-bearing graph concepts with exact alias matches.

    The graph is an optional mined overlay. It is intentionally not allowed to use
    FTS, ontology traversal, or untyped term nodes here: candidate generation must
    remain dictionary-constrained and type-compatible.
    """

    repository: KnowledgeGraphRepositoryPort
    source: str = "kg_exact"
    terminal_on_match: bool = False
    unique_output_short_circuit: bool = False

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]:
        del context_window
        systems = allowed_code_systems(entity_type)
        output: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for system in systems or ():
            nodes = self.repository.search_nodes(
                mention,
                entity_type=entity_type.value,
                code_system=system.value,
                limit=limit,
                exact_only=True,
            )
            for node in nodes:
                if node.code is None or node.code_system != system.value:
                    continue
                key = (node.code_system, node.code)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    semantic_type = EntityType(node.entity_type)
                    code_system = CodeSystem(node.code_system)
                except ValueError:
                    # INVARIANT: an unknown graph enum must never leak into a Candidate.
                    continue
                if semantic_type != entity_type:
                    continue
                output.append(
                    Candidate(
                        concept_id=node.node_id,
                        code=node.code,
                        code_system=code_system,
                        canonical_name=node.label,
                        semantic_type=semantic_type,
                        score=1.0,
                        source=self.source,
                        matched_alias=mention,
                    )
                )
                if len(output) >= limit:
                    return output
        return output


def _candidate(
    entry: ConceptEntry,
    score: float,
    source: str,
    matched_alias: str,
    *,
    reviewed_mapping: bool = False,
) -> Candidate:
    return Candidate(
        concept_id=entry.concept_id,
        code=entry.code,
        code_system=entry.code_system,
        canonical_name=entry.canonical_name,
        semantic_type=entry.semantic_type,
        score=score,
        source=source,
        matched_alias=matched_alias,
        reviewed_mapping=reviewed_mapping,
    )


def _candidate_from_candidate(
    candidate: Candidate,
    score: float,
    source: str,
    matched_alias: str,
) -> Candidate:
    return Candidate(
        concept_id=candidate.concept_id,
        code=candidate.code,
        code_system=candidate.code_system,
        canonical_name=candidate.canonical_name,
        semantic_type=candidate.semantic_type,
        score=score,
        source=source,
        matched_alias=matched_alias,
        reviewed_mapping=candidate.reviewed_mapping,
    )
