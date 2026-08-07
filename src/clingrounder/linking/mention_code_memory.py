"""Cross-fitted mention-to-code memory for dictionary-constrained linking.

The memory captures repeated reviewed assignments without turning one observed occurrence into a
global hard rule. Runtime lookup is genre-aware; evaluation builders can exclude every observation
from the document currently being scored.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.linking.candidate import Candidate
from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.terminology.ports import TerminologyRepository
from clingrounder.utils.io import read_jsonl, write_jsonl
from clingrounder.utils.text import normalize_for_match

__all__ = [
    "CrossFittedMentionCodeMemory",
    "MentionCodeIdentity",
    "MentionCodeMemory",
    "MentionCodeMemoryObservation",
    "MentionCodeMemoryRecord",
    "MentionCodeMemoryRetrieverAdapter",
    "build_cross_fitted_mention_code_memory",
    "build_mention_code_memory",
    "load_mention_code_memory",
    "write_mention_code_memory",
]

_DEFAULT_GENRE = "unknown"


@dataclass(frozen=True, slots=True, order=True)
class MentionCodeIdentity:
    """Dictionary identity counted by a memory record."""

    code_system: CodeSystem
    code: str

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("Mention-code identity requires a non-empty code")


@dataclass(frozen=True, slots=True)
class MentionCodeMemoryObservation:
    """One supervised mention/code assignment with document-level provenance."""

    document_id: str
    mention: str
    entity_type: EntityType
    genre: str
    code_system: CodeSystem
    code: str

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("Mention-code observation requires document_id")
        if not normalize_for_match(self.mention):
            raise ValueError("Mention-code observation requires a non-empty mention")
        if not self.genre.strip():
            raise ValueError("Mention-code observation requires genre")
        if not self.code.strip():
            raise ValueError("Mention-code observation requires code")


@dataclass(frozen=True, slots=True)
class MentionCodeMemoryRecord:
    """Distribution summary for one normalized mention/type/genre key."""

    normalized_mention: str
    entity_type: EntityType
    genre: str
    code_counts: tuple[tuple[MentionCodeIdentity, int], ...]
    document_support: int
    entropy: float
    most_common_probability: float

    def __post_init__(self) -> None:
        if not self.normalized_mention:
            raise ValueError("Mention-code memory requires a normalized mention")
        if not self.genre:
            raise ValueError("Mention-code memory requires genre")
        if self.document_support < 1:
            raise ValueError("Mention-code memory document_support must be positive")
        if not self.code_counts or any(count < 1 for _, count in self.code_counts):
            raise ValueError("Mention-code memory requires positive code counts")
        if not math.isfinite(self.entropy) or self.entropy < 0.0:
            raise ValueError("Mention-code memory entropy must be finite and non-negative")
        if not 0.0 < self.most_common_probability <= 1.0:
            raise ValueError("Mention-code memory probability must be within (0, 1]")

    @property
    def most_common(self) -> MentionCodeIdentity:
        """Return the deterministic mode of this distribution."""

        return self.code_counts[0][0]

    def is_high_confidence(
        self,
        *,
        minimum_document_support: int = 2,
        minimum_probability: float = 0.95,
    ) -> bool:
        """Apply the conservative terminal-memory gate."""

        return (
            self.document_support >= minimum_document_support
            and self.most_common_probability >= minimum_probability
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "normalized_mention": self.normalized_mention,
            "entity_type": self.entity_type.value,
            "genre": self.genre,
            "code_counts": [
                {
                    "code_system": identity.code_system.value,
                    "code": identity.code,
                    "count": count,
                }
                for identity, count in self.code_counts
            ],
            "document_support": self.document_support,
            "entropy": self.entropy,
            "most_common_probability": self.most_common_probability,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "MentionCodeMemoryRecord":
        raw_counts = payload.get("code_counts")
        if not isinstance(raw_counts, list):
            raise ValueError("Mention-code memory code_counts must be a list")
        counts: list[tuple[MentionCodeIdentity, int]] = []
        for raw_count in raw_counts:
            if not isinstance(raw_count, Mapping):
                raise ValueError("Mention-code memory count must be an object")
            count = raw_count.get("count")
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError("Mention-code memory count must be an integer")
            counts.append(
                (
                    MentionCodeIdentity(
                        CodeSystem(str(raw_count.get("code_system", ""))),
                        str(raw_count.get("code", "")),
                    ),
                    count,
                )
            )
        return cls(
            normalized_mention=str(payload.get("normalized_mention", "")),
            entity_type=EntityType(str(payload.get("entity_type", ""))),
            genre=str(payload.get("genre", "")),
            code_counts=tuple(counts),
            document_support=_integer_payload(payload, "document_support"),
            entropy=_float_payload(payload, "entropy"),
            most_common_probability=_float_payload(
                payload, "most_common_probability"
            ),
        )


@dataclass(frozen=True, slots=True)
class MentionCodeMemory:
    """Immutable mention distribution index."""

    records: tuple[MentionCodeMemoryRecord, ...]

    def __post_init__(self) -> None:
        keys = [
            (record.normalized_mention, record.entity_type, record.genre)
            for record in self.records
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Mention-code memory contains duplicate keys")

    @property
    def by_key(self) -> dict[tuple[str, EntityType, str], MentionCodeMemoryRecord]:
        return {
            (record.normalized_mention, record.entity_type, record.genre): record
            for record in self.records
        }

    def lookup(
        self,
        mention: str,
        entity_type: EntityType,
        genre: str,
    ) -> MentionCodeMemoryRecord | None:
        """Look up the exact genre key, then a deliberately explicit unknown fallback."""

        normalized = normalize_for_match(mention)
        index = self.by_key
        return index.get((normalized, entity_type, genre)) or index.get(
            (normalized, entity_type, _DEFAULT_GENRE)
        )


@dataclass(frozen=True, slots=True)
class CrossFittedMentionCodeMemory:
    """Per-fold memories trained without documents assigned to that fold."""

    memories_by_fold: tuple[tuple[int, MentionCodeMemory], ...]
    document_folds: tuple[tuple[str, int], ...]

    def memory_for_document(self, document_id: str) -> MentionCodeMemory:
        fold_by_document = dict(self.document_folds)
        if document_id not in fold_by_document:
            raise KeyError(f"Unknown cross-fit document {document_id!r}")
        return dict(self.memories_by_fold)[fold_by_document[document_id]]


@dataclass(frozen=True, slots=True)
class MentionCodeMemoryRetrieverAdapter:
    """Resolve one memory tier through the canonical terminology repository.

    Instantiate a terminal high-confidence adapter before exact lookup and a non-terminal prior
    adapter after learned lexical sources. Keeping the tiers separate prevents an ambiguous memory
    distribution from short-circuiting stronger terminology evidence.
    """

    memory: MentionCodeMemory
    repository: TerminologyRepository
    genre: str = _DEFAULT_GENRE
    high_confidence_only: bool = True
    minimum_document_support: int = 2
    minimum_probability: float = 0.95
    source: str = "mention_memory"

    @property
    def terminal_on_match(self) -> bool:
        return self.high_confidence_only

    @property
    def unique_output_short_circuit(self) -> bool:
        return False

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]:
        del context_window
        if limit < 1:
            raise ValueError("limit must be at least 1")
        record = self.memory.lookup(mention, entity_type, self.genre)
        if record is None:
            return []
        high_confidence = record.is_high_confidence(
            minimum_document_support=self.minimum_document_support,
            minimum_probability=self.minimum_probability,
        )
        if high_confidence != self.high_confidence_only:
            return []

        output: list[Candidate] = []
        total = sum(count for _, count in record.code_counts)
        for identity, count in record.code_counts:
            entry = self.repository.get_by_code(identity.code_system, identity.code)
            if entry is None or entry.semantic_type is not entity_type:
                continue
            probability = count / total
            output.append(
                Candidate(
                    concept_id=entry.concept_id,
                    code=entry.code,
                    code_system=entry.code_system,
                    canonical_name=entry.canonical_name,
                    semantic_type=entry.semantic_type,
                    score=probability if high_confidence else 0.5 * probability,
                    source=self.source if high_confidence else f"{self.source}_prior",
                    matched_alias=mention,
                )
            )
        return output[:limit]


def build_mention_code_memory(
    observations: Iterable[MentionCodeMemoryObservation],
    *,
    include_genre_fallback: bool = True,
) -> MentionCodeMemory:
    """Aggregate supervised observations into deterministic distribution records."""

    grouped: dict[
        tuple[str, EntityType, str],
        list[MentionCodeMemoryObservation],
    ] = defaultdict(list)
    for observation in observations:
        normalized = normalize_for_match(observation.mention)
        grouped[(normalized, observation.entity_type, observation.genre)].append(observation)
        if include_genre_fallback and observation.genre != _DEFAULT_GENRE:
            grouped[(normalized, observation.entity_type, _DEFAULT_GENRE)].append(observation)

    records: list[MentionCodeMemoryRecord] = []
    for (mention, entity_type, genre), values in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1].value, item[0][2]),
    ):
        counts = Counter(
            MentionCodeIdentity(item.code_system, item.code) for item in values
        )
        total = sum(counts.values())
        probabilities = tuple(count / total for count in counts.values())
        ordered_counts = tuple(
            sorted(
                counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0].code_system.value,
                    item[0].code,
                ),
            )
        )
        records.append(
            MentionCodeMemoryRecord(
                normalized_mention=mention,
                entity_type=entity_type,
                genre=genre,
                code_counts=ordered_counts,
                document_support=len({item.document_id for item in values}),
                entropy=-sum(
                    probability * math.log2(probability)
                    for probability in probabilities
                    if probability > 0.0
                ),
                most_common_probability=ordered_counts[0][1] / total,
            )
        )
    return MentionCodeMemory(tuple(records))


def write_mention_code_memory(
    memory: MentionCodeMemory,
    path: str | Path,
) -> None:
    """Write a deterministic JSONL artifact; callers own source fingerprints in a manifest."""

    write_jsonl(path, (record.to_json() for record in memory.records))


def load_mention_code_memory(path: str | Path) -> MentionCodeMemory:
    """Load and validate a prebuilt mention-code memory artifact."""

    return MentionCodeMemory(
        tuple(MentionCodeMemoryRecord.from_json(row) for row in read_jsonl(path))
    )


def build_cross_fitted_mention_code_memory(
    observations: Iterable[MentionCodeMemoryObservation],
    document_folds: Mapping[str, int],
) -> CrossFittedMentionCodeMemory:
    """Build one memory per held-out fold with document-grouped exclusion."""

    values = tuple(observations)
    observed_documents = {item.document_id for item in values}
    missing = observed_documents - set(document_folds)
    if missing:
        raise ValueError(f"Cross-fit fold mapping misses documents: {sorted(missing)}")
    if any(fold < 0 for fold in document_folds.values()):
        raise ValueError("Cross-fit fold identifiers must be non-negative")
    folds = sorted({document_folds[document_id] for document_id in observed_documents})
    if len(folds) < 2:
        raise ValueError("Cross-fitted mention memory requires at least two folds")
    memories = tuple(
        (
            fold,
            build_mention_code_memory(
                item for item in values if document_folds[item.document_id] != fold
            ),
        )
        for fold in folds
    )
    # INVARIANT: no observation from a held-out document enters its evaluation memory.
    return CrossFittedMentionCodeMemory(
        memories_by_fold=memories,
        document_folds=tuple(sorted(document_folds.items())),
    )


def _integer_payload(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Mention-code memory {field} must be an integer")
    return value


def _float_payload(payload: Mapping[str, object], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Mention-code memory {field} must be numeric")
    return float(value)
