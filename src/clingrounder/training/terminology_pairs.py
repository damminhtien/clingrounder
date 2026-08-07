"""Deterministic same-concept synonym pairs for terminology metric learning.

The record contract follows the useful part of SapBERT's curriculum: two surfaces
are positive only when they belong to the same pinned concept. It intentionally
does not depend on SapBERT code, UMLS, or a specific model implementation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from clingrounder.dictionaries.synonym_table import ConceptEntry
from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.utils.text import normalize_for_match

__all__ = [
    "SynonymPairMode",
    "TerminologyPairConfig",
    "TerminologySynonymPair",
    "build_terminology_synonym_pairs",
    "write_terminology_pair_dataset",
]


class SynonymPairMode(StrEnum):
    """Pair expansion policy."""

    CANONICAL_TO_ALIAS = "canonical_to_alias"
    ALL_PAIRS = "all_pairs"


@dataclass(frozen=True, slots=True)
class TerminologyPairConfig:
    """Bounds for deterministic positive-pair expansion."""

    mode: SynonymPairMode = SynonymPairMode.CANONICAL_TO_ALIAS
    max_names_per_concept: int = 16
    max_pairs_per_concept: int = 32
    include_abbreviations: bool = True

    def __post_init__(self) -> None:
        if self.max_names_per_concept < 2:
            raise ValueError("max_names_per_concept must be at least 2.")
        if self.max_pairs_per_concept < 1:
            raise ValueError("max_pairs_per_concept must be at least 1.")


@dataclass(frozen=True, slots=True)
class TerminologySynonymPair:
    """One positive pair with terminology provenance and type constraints."""

    pair_id: str
    concept_id: str
    code: str | None
    code_system: CodeSystem
    entity_type: EntityType
    left: str
    right: str
    left_role: str
    right_role: str
    source: str

    def to_json(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "concept_id": self.concept_id,
            "code": self.code,
            "code_system": self.code_system.value,
            "entity_type": self.entity_type.value,
            "left": self.left,
            "right": self.right,
            "left_role": self.left_role,
            "right_role": self.right_role,
            "source": self.source,
            "label": 1,
        }


@dataclass(frozen=True, slots=True)
class _TermSurface:
    text: str
    role: str


def build_terminology_synonym_pairs(
    entries: Iterable[ConceptEntry],
    *,
    config: TerminologyPairConfig = TerminologyPairConfig(),
) -> tuple[TerminologySynonymPair, ...]:
    """Build bounded positive pairs without crossing concepts or code systems."""

    output: list[TerminologySynonymPair] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for entry in sorted(
        entries,
        key=lambda item: (
            item.code_system.value,
            item.semantic_type.value,
            item.concept_id,
        ),
    ):
        surfaces = _surfaces(entry, config)[: config.max_names_per_concept]
        if len(surfaces) < 2:
            continue
        candidates = _pair_candidates(surfaces, config.mode)
        concept_count = 0
        for left, right in candidates:
            normalized_left = normalize_for_match(left.text)
            normalized_right = normalize_for_match(right.text)
            ordered = (
                min(normalized_left, normalized_right),
                max(normalized_left, normalized_right),
            )
            key = (entry.concept_id, ordered[0], ordered[1])
            if not all(ordered) or ordered[0] == ordered[1] or key in seen_pairs:
                continue
            seen_pairs.add(key)
            output.append(_pair(entry, left, right, ordered))
            concept_count += 1
            # SCALING: terminology concepts can have thousands of aliases. The cap
            # prevents a quadratic training artifact while keeping generation stable.
            if concept_count >= config.max_pairs_per_concept:
                break
    return tuple(output)


def write_terminology_pair_dataset(
    pairs: Iterable[TerminologySynonymPair],
    output_path: str | Path,
    *,
    config: TerminologyPairConfig,
    source_fingerprints: dict[str, str],
) -> dict[str, Any]:
    """Atomically write JSONL plus a fingerprinted manifest."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = tuple(pairs)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for pair in rows:
            handle.write(json.dumps(pair.to_json(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, output)

    dataset_sha256 = _sha256_file(output)
    manifest = {
        "schema_version": "terminology-synonym-pairs.v1",
        "record_count": len(rows),
        "dataset_path": output.name,
        "dataset_sha256": dataset_sha256,
        "source_fingerprints": dict(sorted(source_fingerprints.items())),
        "config": {
            "mode": config.mode.value,
            "max_names_per_concept": config.max_names_per_concept,
            "max_pairs_per_concept": config.max_pairs_per_concept,
            "include_abbreviations": config.include_abbreviations,
        },
    }
    manifest_path = output.with_suffix(f"{output.suffix}.manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _surfaces(
    entry: ConceptEntry,
    config: TerminologyPairConfig,
) -> tuple[_TermSurface, ...]:
    values: list[tuple[str | None, str]] = [
        (entry.canonical_name, "canonical"),
        (entry.official_name_vi, "official_vi"),
        (entry.official_name_en, "official_en"),
        *((value, "alias") for value in entry.aliases),
        *((value, "synonym") for value in entry.synonyms),
    ]
    if config.include_abbreviations:
        values.extend((value, "abbreviation") for value in entry.abbreviations)
    values.extend(
        (
            (entry.ingredient, "ingredient"),
            (entry.brand_name, "brand"),
            (entry.generic_name, "generic"),
        )
    )

    blocked = {normalize_for_match(value) for value in entry.blocked_aliases}
    output: list[_TermSurface] = []
    seen: set[str] = set()
    for value, role in values:
        text = value.strip() if value else ""
        normalized = normalize_for_match(text)
        if not normalized or normalized in blocked or normalized in seen:
            continue
        seen.add(normalized)
        output.append(_TermSurface(text=text, role=role))
    return tuple(output)


def _pair_candidates(
    surfaces: tuple[_TermSurface, ...],
    mode: SynonymPairMode,
) -> Iterable[tuple[_TermSurface, _TermSurface]]:
    if mode == SynonymPairMode.CANONICAL_TO_ALIAS:
        return ((surfaces[0], surface) for surface in surfaces[1:])
    return itertools.combinations(surfaces, 2)


def _pair(
    entry: ConceptEntry,
    left: _TermSurface,
    right: _TermSurface,
    normalized_order: tuple[str, str],
) -> TerminologySynonymPair:
    digest_payload = "\0".join(
        (
            entry.code_system.value,
            entry.semantic_type.value,
            entry.concept_id,
            *normalized_order,
        )
    ).encode("utf-8")
    pair_id = hashlib.sha256(digest_payload).hexdigest()[:20]
    return TerminologySynonymPair(
        pair_id=pair_id,
        concept_id=entry.concept_id,
        code=entry.code,
        code_system=entry.code_system,
        entity_type=entry.semantic_type,
        left=left.text,
        right=right.text,
        left_role=left.role,
        right_role=right.role,
        source=entry.source,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
