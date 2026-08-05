"""Versioned synonym-vector indexes for dense terminology retrieval.

Concept surfaces are encoded once with a pinned model. Runtime search is sharded by entity type
and code system, so dense retrieval cannot recover an incompatible code and filter it only after
the global top-k has already discarded useful candidates.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.retrieval.dense_retriever import DenseHit, TextEncoderPort
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "FaissSynonymVectorIndex",
    "InMemorySynonymVectorIndex",
    "SynonymIndexMetadata",
    "SynonymVectorRecord",
    "build_synonym_vector_records",
    "fingerprint_terminology_entries",
    "write_faiss_synonym_index",
]

_SCHEMA_VERSION = "synonym-vector-index.v1"


@dataclass(frozen=True, slots=True)
class SynonymIndexMetadata:
    """Provenance required to load a derived vector index."""

    model_id: str
    revision: str
    terminology_fingerprint: str
    dimensions: int
    vector_count: int
    concept_count: int
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name, value in (
            ("model_id", self.model_id),
            ("revision", self.revision),
            ("terminology_fingerprint", self.terminology_fingerprint),
        ):
            if not value.strip():
                raise ValueError(f"Synonym index {field_name} must be non-empty")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported synonym index schema {self.schema_version!r}")
        if self.dimensions < 1 or self.vector_count < 1 or self.concept_count < 1:
            raise ValueError("Synonym index dimensions and counts must be positive")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "terminology_fingerprint": self.terminology_fingerprint,
            "dimensions": self.dimensions,
            "vector_count": self.vector_count,
            "concept_count": self.concept_count,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "SynonymIndexMetadata":
        return cls(
            model_id=str(payload.get("model_id", "")),
            revision=str(payload.get("revision", "")),
            terminology_fingerprint=str(payload.get("terminology_fingerprint", "")),
            dimensions=_integer_field(payload, "dimensions"),
            vector_count=_integer_field(payload, "vector_count"),
            concept_count=_integer_field(payload, "concept_count"),
            schema_version=str(payload.get("schema_version", "")),
        )


@dataclass(frozen=True, slots=True)
class SynonymVectorRecord:
    """One normalized synonym embedding tied to a terminology concept."""

    concept_id: str
    entity_type: EntityType
    code_system: CodeSystem
    surface: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.concept_id.strip() or not normalize_for_match(self.surface):
            raise ValueError("Synonym vector requires concept_id and surface")
        if not self.vector or any(not math.isfinite(value) for value in self.vector):
            raise ValueError("Synonym vector must contain finite values")


@dataclass(frozen=True, slots=True)
class InMemorySynonymVectorIndex:
    """Correctness backend for small corpora and backend contract tests."""

    records: tuple[SynonymVectorRecord, ...]

    def search(
        self,
        vector: Sequence[float],
        *,
        entity_type: EntityType,
        code_systems: Sequence[CodeSystem] | None,
        limit: int,
    ) -> list[DenseHit]:
        if limit < 1:
            raise ValueError("Dense search limit must be positive")
        query = _unit_vector(vector)
        systems = set(code_systems) if code_systems is not None else None
        # INVARIANT: metadata filtering happens before ranking and LIMIT.
        best_by_concept: dict[str, float] = {}
        for record in self.records:
            if record.entity_type is not entity_type:
                continue
            if systems is not None and record.code_system not in systems:
                continue
            similarity = sum(left * right for left, right in zip(query, record.vector, strict=True))
            score = _bounded_cosine_score(similarity)
            best_by_concept[record.concept_id] = max(
                score,
                best_by_concept.get(record.concept_id, 0.0),
            )
        return [
            DenseHit(concept_id, score)
            for concept_id, score in sorted(
                best_by_concept.items(),
                key=lambda item: (-item[1], item[0]),
            )[:limit]
        ]


class FaissSynonymVectorIndex:
    """Read-only FAISS IndexFlatIP shards with pinned model/source provenance."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_model_id: str,
        expected_revision: str,
        expected_terminology_fingerprint: str,
    ) -> None:
        self.path = Path(path)
        payload = _read_json_object(self.path / "manifest.json")
        metadata_payload = payload.get("metadata")
        if not isinstance(metadata_payload, dict):
            raise ValueError("Synonym index manifest requires metadata")
        self.metadata = SynonymIndexMetadata.from_json(metadata_payload)
        expected = (
            expected_model_id,
            expected_revision,
            expected_terminology_fingerprint,
        )
        actual = (
            self.metadata.model_id,
            self.metadata.revision,
            self.metadata.terminology_fingerprint,
        )
        if actual != expected:
            raise ValueError("Synonym index model/revision/terminology fingerprint mismatch")
        raw_shards = payload.get("shards")
        if not isinstance(raw_shards, list):
            raise ValueError("Synonym index manifest requires shard records")
        self._shards = tuple(_validated_shard(item) for item in raw_shards)
        self._loaded: dict[str, tuple[Any, tuple[str, ...]]] = {}

    def search(
        self,
        vector: Sequence[float],
        *,
        entity_type: EntityType,
        code_systems: Sequence[CodeSystem] | None,
        limit: int,
    ) -> list[DenseHit]:
        if limit < 1:
            raise ValueError("Dense search limit must be positive")
        faiss, numpy = _faiss_dependencies()
        query = numpy.asarray([_unit_vector(vector)], dtype="float32")
        systems = set(code_systems) if code_systems is not None else None
        best_by_concept: dict[str, float] = {}
        for shard in self._shards:
            if shard["entity_type"] != entity_type.value:
                continue
            if systems is not None and CodeSystem(shard["code_system"]) not in systems:
                continue
            index, concept_ids = self._load_shard(shard, faiss)
            scores, rows = index.search(query, min(limit, len(concept_ids)))
            for similarity, row in zip(scores[0], rows[0], strict=True):
                if int(row) < 0:
                    continue
                concept_id = concept_ids[int(row)]
                score = _bounded_cosine_score(float(similarity))
                best_by_concept[concept_id] = max(
                    score,
                    best_by_concept.get(concept_id, 0.0),
                )
        return [
            DenseHit(concept_id, score)
            for concept_id, score in sorted(
                best_by_concept.items(),
                key=lambda item: (-item[1], item[0]),
            )[:limit]
        ]

    def close(self) -> None:
        """Release loaded FAISS shard objects while retaining manifest metadata."""

        self._loaded.clear()

    def _load_shard(
        self,
        shard: dict[str, str],
        faiss: Any,
    ) -> tuple[Any, tuple[str, ...]]:
        name = shard["name"]
        cached = self._loaded.get(name)
        if cached is not None:
            return cached
        index = faiss.read_index(str(self.path / f"{name}.faiss"))
        concept_payload = _read_json_object(self.path / f"{name}.json")
        raw_ids = concept_payload.get("concept_ids")
        if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
            raise ValueError(f"Invalid synonym index concept IDs for shard {name}")
        if index.ntotal != len(raw_ids):
            raise ValueError(f"Synonym index row count mismatch for shard {name}")
        loaded = index, tuple(raw_ids)
        self._loaded[name] = loaded
        return loaded


def build_synonym_vector_records(
    entries: Iterable[ConceptEntry],
    encoder: TextEncoderPort,
    *,
    batch_size: int = 128,
) -> tuple[SynonymVectorRecord, ...]:
    """Encode every unique allowed concept surface with one reusable text encoder."""

    if batch_size < 1:
        raise ValueError("Synonym encoding batch_size must be positive")
    surfaces: list[tuple[ConceptEntry, str]] = []
    for entry in sorted(
        entries,
        key=lambda item: (item.semantic_type.value, item.code_system.value, item.concept_id),
    ):
        seen: set[str] = set()
        for surface in entry.all_names:
            normalized = normalize_for_match(surface)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            surfaces.append((entry, surface))
    records: list[SynonymVectorRecord] = []
    for start in range(0, len(surfaces), batch_size):
        batch = surfaces[start : start + batch_size]
        vectors = encoder.encode(tuple(surface for _, surface in batch))
        if len(vectors) != len(batch):
            raise ValueError("Text encoder returned the wrong synonym-vector count")
        records.extend(
            SynonymVectorRecord(
                concept_id=entry.concept_id,
                entity_type=entry.semantic_type,
                code_system=entry.code_system,
                surface=surface,
                vector=_unit_vector(vector),
            )
            for (entry, surface), vector in zip(batch, vectors, strict=True)
        )
    if not records:
        raise ValueError("Cannot build a synonym index without terminology surfaces")
    dimensions = len(records[0].vector)
    if any(len(record.vector) != dimensions for record in records):
        raise ValueError("Text encoder returned inconsistent vector dimensions")
    return tuple(records)


def write_faiss_synonym_index(
    records: Iterable[SynonymVectorRecord],
    output_path: str | Path,
    *,
    model_id: str,
    revision: str,
    terminology_fingerprint: str,
) -> SynonymIndexMetadata:
    """Atomically write immutable FAISS shards and their provenance manifest."""

    values = tuple(records)
    if not values:
        raise ValueError("Cannot write an empty synonym index")
    dimensions = len(values[0].vector)
    if any(len(item.vector) != dimensions for item in values):
        raise ValueError("Synonym index vectors have inconsistent dimensions")
    metadata = SynonymIndexMetadata(
        model_id=model_id,
        revision=revision,
        terminology_fingerprint=terminology_fingerprint,
        dimensions=dimensions,
        vector_count=len(values),
        concept_count=len({item.concept_id for item in values}),
    )
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Synonym index path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        faiss, numpy = _faiss_dependencies()
        grouped: dict[tuple[EntityType, CodeSystem], list[SynonymVectorRecord]] = defaultdict(list)
        for record in values:
            grouped[(record.entity_type, record.code_system)].append(record)
        shards: list[dict[str, str]] = []
        for (entity_type, code_system), shard_records in sorted(
            grouped.items(),
            key=lambda item: (item[0][0].value, item[0][1].value),
        ):
            name = hashlib.sha256(
                f"{entity_type.value}\x1f{code_system.value}".encode()
            ).hexdigest()[:16]
            matrix = numpy.asarray([item.vector for item in shard_records], dtype="float32")
            index = faiss.IndexFlatIP(dimensions)
            index.add(matrix)
            faiss.write_index(index, str(temporary / f"{name}.faiss"))
            (temporary / f"{name}.json").write_text(
                json.dumps(
                    {"concept_ids": [item.concept_id for item in shard_records]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            shards.append(
                {
                    "name": name,
                    "entity_type": entity_type.value,
                    "code_system": code_system.value,
                }
            )
        (temporary / "manifest.json").write_text(
            json.dumps(
                {"metadata": metadata.to_json(), "shards": shards},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def fingerprint_terminology_entries(entries: Iterable[ConceptEntry]) -> str:
    """Hash concept identities and surfaces independently from file layout."""

    digest = hashlib.sha256()
    for entry in sorted(
        entries,
        key=lambda item: (item.code_system.value, item.concept_id),
    ):
        payload = {
            "concept_id": entry.concept_id,
            "code": entry.code,
            "code_system": entry.code_system.value,
            "entity_type": entry.semantic_type.value,
            "names": entry.all_names,
        }
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _unit_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Dense vectors must contain finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("Dense vectors must have non-zero norm")
    return tuple(value / norm for value in values)


def _bounded_cosine_score(similarity: float) -> float:
    return min(1.0, max(0.0, (similarity + 1.0) / 2.0))


def _faiss_dependencies() -> tuple[Any, Any]:
    try:
        faiss = importlib.import_module("faiss")
        numpy = importlib.import_module("numpy")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FAISS synonym indexes require the 'retrieval' extra: "
            "pip install 'medical-kg-nlp[retrieval]'"
        ) from exc
    return faiss, numpy


def _integer_field(payload: dict[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Synonym index {field} must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Synonym index {field} must be an integer") from exc


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return {str(key): value for key, value in payload.items()}


def _validated_shard(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Synonym index shard must be an object")
    result = {field: str(payload.get(field, "")) for field in ("name", "entity_type", "code_system")}
    if not all(result.values()):
        raise ValueError("Synonym index shard fields must be non-empty")
    EntityType(result["entity_type"])
    CodeSystem(result["code_system"])
    return result
