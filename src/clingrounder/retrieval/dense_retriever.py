"""Storage-neutral dense retrieval over a separately built vector index."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from clingrounder.linking.candidate import Candidate
from clingrounder.retrieval.constraints import allowed_code_systems
from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.terminology.ports import TerminologyRepository

__all__ = [
    "DenseHit",
    "DenseRetrieverAdapter",
    "DenseVectorIndexPort",
    "TextEncoderPort",
]


class TextEncoderPort(Protocol):
    """Encode text without exposing a model framework to retrieval orchestration."""

    def encode(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


@dataclass(frozen=True)
class DenseHit:
    """One normalized similarity result from a vector index."""

    concept_id: str
    score: float

    def __post_init__(self) -> None:
        if not self.concept_id.strip():
            raise ValueError("Dense hit concept_id must be non-empty")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("Dense hit score must be finite and between 0 and 1")


class DenseVectorIndexPort(Protocol):
    """Search a versioned vector index already filtered by terminology metadata."""

    def search(
        self,
        vector: Sequence[float],
        *,
        entity_type: EntityType,
        code_systems: Sequence[CodeSystem] | None,
        limit: int,
    ) -> list[DenseHit]: ...


@dataclass(frozen=True)
class DenseRetrieverAdapter:
    """Resolve dense index hits through the canonical terminology repository."""

    encoder: TextEncoderPort
    index: DenseVectorIndexPort
    repository: TerminologyRepository
    source: str = "dense"
    terminal_on_match: bool = False
    unique_output_short_circuit: bool = False

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]:
        """Encode a mention and enforce type/system constraints after index lookup."""

        del context_window
        vectors = self.encoder.encode((mention,))
        if len(vectors) != 1:
            raise ValueError("Text encoder must return exactly one vector for one mention")
        systems = allowed_code_systems(entity_type)
        hits = self.index.search(
            vectors[0],
            entity_type=entity_type,
            code_systems=systems,
            limit=limit,
        )
        output: list[Candidate] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.concept_id in seen:
                continue
            entry = self.repository.get_by_concept_id(hit.concept_id)
            if entry is None or entry.semantic_type != entity_type:
                continue
            if systems is not None and entry.code_system not in systems:
                continue
            seen.add(hit.concept_id)
            output.append(
                Candidate(
                    concept_id=entry.concept_id,
                    code=entry.code,
                    code_system=entry.code_system,
                    canonical_name=entry.canonical_name,
                    semantic_type=entry.semantic_type,
                    score=hit.score,
                    source=self.source,
                    matched_alias=mention,
                )
            )
        # INVARIANT: a dense hit never bypasses terminology type/code-system constraints.
        return output[:limit]

    def close(self) -> None:
        """Close the encoder and vector index if either owns external resources."""

        for resource in (self.encoder, self.index):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
