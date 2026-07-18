"""Build a versioned SQLite FTS5 index from canonical terminology JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.merge import merge_concept_entries
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.preprocessing.normalizer import NORMALIZATION_CONTRACT_VERSION
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "TERMINOLOGY_INDEX_SCHEMA_VERSION",
    "TerminologyIndexManifest",
    "build_terminology_index",
    "input_fingerprint",
    "source_fingerprint",
    "terminology_cache_path",
]

TERMINOLOGY_INDEX_SCHEMA_VERSION = "sqlite-fts5-v2"


@dataclass(frozen=True)
class TerminologyIndexManifest:
    """Reproducibility metadata emitted for one derived index."""

    index_path: str
    schema_version: str
    normalization_version: str
    source_fingerprint: str
    alias_overlay_fingerprint: str
    input_fingerprint: str
    concept_count: int
    alias_count: int
    overlay_alias_count: int

    def to_json(self) -> dict[str, str | int]:
        """Return a JSON-serializable manifest payload."""

        return asdict(self)


def source_fingerprint(source_paths: tuple[str | Path, ...]) -> str:
    """Hash source bytes in declared merge order."""

    if not source_paths:
        raise ValueError("At least one terminology source is required")
    return _paths_fingerprint(source_paths, require_non_empty=True)


def input_fingerprint(
    source_paths: tuple[str | Path, ...],
    alias_overlay_paths: tuple[str | Path, ...] = (),
) -> str:
    """Hash canonical sources and overlays with explicit domain separation."""

    digest = hashlib.sha256()
    digest.update(b"canonical-sources\0")
    digest.update(source_fingerprint(source_paths).encode("ascii"))
    digest.update(b"alias-overlays\0")
    digest.update(_paths_fingerprint(alias_overlay_paths).encode("ascii"))
    return digest.hexdigest()


def terminology_cache_path(
    cache_dir: str | Path,
    source_paths: tuple[str | Path, ...],
    *,
    alias_overlay_paths: tuple[str | Path, ...] = (),
    normalization_version: str = NORMALIZATION_CONTRACT_VERSION,
    schema_version: str = TERMINOLOGY_INDEX_SCHEMA_VERSION,
) -> Path:
    """Derive a content-addressed cache path for the source and schema contract."""

    digest = hashlib.sha256()
    digest.update(input_fingerprint(source_paths, alias_overlay_paths).encode("ascii"))
    digest.update(schema_version.encode("utf-8"))
    digest.update(normalization_version.encode("utf-8"))
    return Path(cache_dir) / f"terminology-{digest.hexdigest()[:20]}.sqlite3"


def build_terminology_index(
    source_paths: tuple[str | Path, ...],
    *,
    alias_overlay_paths: tuple[str | Path, ...] = (),
    output_path: str | Path | None = None,
    cache_dir: str | Path = ".cache/medical-kg/terminology",
    normalization_version: str = NORMALIZATION_CONTRACT_VERSION,
) -> TerminologyIndexManifest:
    """Build into a sibling temporary file and atomically publish on success."""

    canonical_fingerprint = source_fingerprint(source_paths)
    overlay_fingerprint = _paths_fingerprint(alias_overlay_paths)
    combined_fingerprint = input_fingerprint(source_paths, alias_overlay_paths)
    target = (
        Path(output_path)
        if output_path is not None
        else terminology_cache_path(
            cache_dir,
            source_paths,
            alias_overlay_paths=alias_overlay_paths,
            normalization_version=normalization_version,
        )
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    entries, overlay_sources, overlay_alias_count = _load_entries(
        source_paths,
        alias_overlay_paths,
    )
    try:
        connection = sqlite3.connect(temporary)
        try:
            _initialize_schema(connection)
            alias_count = _write_entries(
                connection,
                entries,
                source_paths=source_paths,
                alias_overlay_paths=alias_overlay_paths,
                source_fingerprint_value=canonical_fingerprint,
                alias_overlay_fingerprint_value=overlay_fingerprint,
                input_fingerprint_value=combined_fingerprint,
                normalization_version=normalization_version,
                overlay_sources=overlay_sources,
            )
        finally:
            connection.close()
        # SCALING: readers either observe the previous complete index or this complete index.
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return TerminologyIndexManifest(
        index_path=str(target),
        schema_version=TERMINOLOGY_INDEX_SCHEMA_VERSION,
        normalization_version=normalization_version,
        source_fingerprint=canonical_fingerprint,
        alias_overlay_fingerprint=overlay_fingerprint,
        input_fingerprint=combined_fingerprint,
        concept_count=len(entries),
        alias_count=alias_count,
        overlay_alias_count=overlay_alias_count,
    )


def _load_entries(
    source_paths: tuple[str | Path, ...],
    alias_overlay_paths: tuple[str | Path, ...],
) -> tuple[list[ConceptEntry], dict[tuple[str, str], str], int]:
    entries: list[ConceptEntry] = []
    for source_path in source_paths:
        entries.extend(DictionaryStore.load_entries_jsonl(source_path))
    merged = merge_concept_entries(entries)
    return _apply_alias_overlays(merged, alias_overlay_paths)


def _apply_alias_overlays(
    entries: list[ConceptEntry],
    alias_overlay_paths: tuple[str | Path, ...],
) -> tuple[list[ConceptEntry], dict[tuple[str, str], str], int]:
    by_concept_id = {entry.concept_id: entry for entry in entries}
    aliases_by_concept: dict[str, list[str]] = {}
    overlay_sources: dict[tuple[str, str], str] = {}
    normalized_targets: dict[tuple[str, EntityType, CodeSystem], str] = {}
    base_targets: dict[tuple[str, EntityType, CodeSystem], set[str]] = {}
    for entry in entries:
        for name in entry.all_names:
            key = (
                normalize_for_match(name),
                entry.semantic_type,
                entry.code_system,
            )
            base_targets.setdefault(key, set()).add(entry.concept_id)

    for overlay_path in alias_overlay_paths:
        path = Path(overlay_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                target, alias = _validated_overlay_row(
                    row,
                    path=path,
                    line_number=line_number,
                    concepts=by_concept_id,
                )
                normalized = normalize_for_match(alias)
                # INVARIANT: runtime exact lookup filters semantic type and code system before
                # LIMIT. Homonyms across those spaces are valid; ambiguity inside one space is not.
                collision_key = (
                    normalized,
                    target.semantic_type,
                    target.code_system,
                )
                existing_targets = base_targets.get(collision_key, set())
                if existing_targets and target.concept_id not in existing_targets:
                    raise ValueError(
                        f"{path}:{line_number}: alias {alias!r} already belongs to canonical "
                        f"concepts {sorted(existing_targets)!r}, not {target.concept_id!r}"
                    )
                previous_target = normalized_targets.get(collision_key)
                if previous_target is not None and previous_target != target.concept_id:
                    raise ValueError(
                        f"{path}:{line_number}: normalized alias {normalized!r} targets both "
                        f"{previous_target!r} and {target.concept_id!r}"
                    )
                normalized_targets[collision_key] = target.concept_id
                existing = {normalize_for_match(value) for value in target.all_names}
                existing.update(
                    normalize_for_match(value)
                    for value in aliases_by_concept.get(target.concept_id, ())
                )
                if normalized in existing:
                    continue
                aliases_by_concept.setdefault(target.concept_id, []).append(alias)
                overlay_sources[(target.concept_id, normalized)] = _overlay_source(row, path)

    if not aliases_by_concept:
        return entries, overlay_sources, 0
    updated = [
        replace(
            entry,
            aliases=(*entry.aliases, *aliases_by_concept.get(entry.concept_id, ())),
        )
        for entry in entries
    ]
    return updated, overlay_sources, sum(map(len, aliases_by_concept.values()))


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE concepts (
            concept_id TEXT PRIMARY KEY,
            code TEXT,
            code_system TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            semantic_type TEXT NOT NULL,
            tty TEXT,
            parent_code TEXT,
            parents_json TEXT NOT NULL,
            aliases_json TEXT NOT NULL,
            synonyms_json TEXT NOT NULL,
            abbreviations_json TEXT NOT NULL,
            blocked_aliases_json TEXT NOT NULL,
            official_name_vi TEXT,
            official_name_en TEXT,
            source TEXT NOT NULL,
            rxnorm_id TEXT,
            ingredient TEXT,
            brand_name TEXT,
            generic_name TEXT,
            dose_form TEXT,
            strength TEXT
        ) WITHOUT ROWID;
        CREATE INDEX concepts_type_system_idx
            ON concepts(semantic_type, code_system, concept_id);
        CREATE INDEX concepts_code_idx
            ON concepts(code_system, code);
        CREATE TABLE aliases (
            alias_id INTEGER PRIMARY KEY,
            surface TEXT NOT NULL,
            normalized TEXT NOT NULL,
            toneless TEXT NOT NULL,
            source TEXT NOT NULL,
            concept_id TEXT NOT NULL REFERENCES concepts(concept_id)
        );
        CREATE INDEX aliases_normalized_idx ON aliases(normalized, concept_id);
        CREATE INDEX aliases_toneless_idx ON aliases(toneless, concept_id);
        CREATE INDEX aliases_concept_idx ON aliases(concept_id);
        CREATE VIRTUAL TABLE aliases_fts USING fts5(
            surface,
            normalized,
            toneless,
            concept_id UNINDEXED,
            tokenize='trigram'
        );
        """
    )


def _write_entries(
    connection: sqlite3.Connection,
    entries: list[ConceptEntry],
    *,
    source_paths: tuple[str | Path, ...],
    alias_overlay_paths: tuple[str | Path, ...],
    source_fingerprint_value: str,
    alias_overlay_fingerprint_value: str,
    input_fingerprint_value: str,
    normalization_version: str,
    overlay_sources: dict[tuple[str, str], str],
) -> int:
    metadata = {
        "schema_version": TERMINOLOGY_INDEX_SCHEMA_VERSION,
        "normalization_version": normalization_version,
        "source_fingerprint": source_fingerprint_value,
        "alias_overlay_fingerprint": alias_overlay_fingerprint_value,
        "input_fingerprint": input_fingerprint_value,
        "source_paths": json.dumps([str(Path(path)) for path in source_paths]),
        "alias_overlay_paths": json.dumps(
            [str(Path(path)) for path in alias_overlay_paths]
        ),
        "concept_count": str(len(entries)),
    }
    with connection:
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        connection.executemany(
            """
            INSERT INTO concepts VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [_concept_row(entry) for entry in entries],
        )
        alias_rows = [
            (surface, normalized, toneless, source, entry.concept_id)
            for entry in entries
            for surface, normalized, toneless, source in _alias_rows(
                entry,
                overlay_sources,
            )
        ]
        connection.executemany(
            """
            INSERT INTO aliases(surface, normalized, toneless, source, concept_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            alias_rows,
        )
        connection.executemany(
            """
            INSERT INTO aliases_fts(surface, normalized, toneless, concept_id)
            VALUES (?, ?, ?, ?)
            """,
            [
                (surface, normalized, toneless, concept_id)
                for surface, normalized, toneless, _, concept_id in alias_rows
            ],
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('alias_count', ?)",
            (str(len(alias_rows)),),
        )
    return len(alias_rows)


def _concept_row(entry: ConceptEntry) -> tuple[object, ...]:
    return (
        entry.concept_id,
        entry.code,
        entry.code_system.value,
        entry.canonical_name,
        entry.semantic_type.value,
        entry.rxnorm_tty,
        entry.parent_code,
        json.dumps(entry.parents, ensure_ascii=False),
        json.dumps(entry.aliases, ensure_ascii=False),
        json.dumps(entry.synonyms, ensure_ascii=False),
        json.dumps(entry.abbreviations, ensure_ascii=False),
        json.dumps(entry.blocked_aliases, ensure_ascii=False),
        entry.official_name_vi,
        entry.official_name_en,
        entry.source,
        entry.rxnorm_id,
        entry.ingredient,
        entry.brand_name,
        entry.generic_name,
        entry.dose_form,
        entry.strength,
    )


def _alias_rows(
    entry: ConceptEntry,
    overlay_sources: dict[tuple[str, str], str],
) -> list[tuple[str, str, str, str]]:
    sources = (
        (entry.canonical_name, "canonical_name"),
        (entry.official_name_vi, "official_name_vi"),
        (entry.official_name_en, "official_name_en"),
        *((value, "alias") for value in entry.aliases),
        *((value, "synonym") for value in entry.synonyms),
        *((value, "abbreviation") for value in entry.abbreviations),
        (entry.ingredient, "ingredient"),
        (entry.brand_name, "brand_name"),
        (entry.generic_name, "generic_name"),
    )
    blocked = {value.casefold().strip() for value in entry.blocked_aliases}
    output: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for value, source in sources:
        if value is None:
            continue
        surface = value.strip()
        key = surface.casefold()
        if not surface or key in blocked or key in seen:
            continue
        seen.add(key)
        normalized = normalize_for_match(surface)
        output.append(
            (
                surface,
                normalized,
                normalize_for_match(surface, strip_diacritics=True),
                overlay_sources.get((entry.concept_id, normalized), source),
            )
        )
    return output


def _validated_overlay_row(
    row: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    concepts: dict[str, ConceptEntry],
) -> tuple[ConceptEntry, str]:
    target_concept_id = str(row.get("target_concept_id", "")).strip()
    alias = str(row.get("alias", "")).strip()
    if not target_concept_id or not alias:
        raise ValueError(
            f"{path}:{line_number}: target_concept_id and alias are required"
        )
    target = concepts.get(target_concept_id)
    if target is None:
        raise ValueError(
            f"{path}:{line_number}: unknown target concept {target_concept_id!r}"
        )
    checks: tuple[tuple[str, object, object], ...] = (
        ("code", row.get("code"), target.code),
        ("code_system", row.get("code_system"), target.code_system.value),
        ("semantic_type", row.get("semantic_type"), target.semantic_type.value),
    )
    for field_name, provided, expected in checks:
        if provided is not None and str(provided) != str(expected):
            raise ValueError(
                f"{path}:{line_number}: {field_name} {provided!r} does not match "
                f"target concept value {expected!r}"
            )
    # Parse enum-bearing fields even when a malformed value happens to stringify like input.
    if row.get("code_system") is not None:
        CodeSystem(str(row["code_system"]))
    if row.get("semantic_type") is not None:
        EntityType(str(row["semantic_type"]))
    if not normalize_for_match(alias):
        raise ValueError(f"{path}:{line_number}: alias normalizes to an empty value")
    return target, alias


def _overlay_source(row: dict[str, Any], path: Path) -> str:
    source = str(row.get("source", "")).strip() or path.name
    return f"overlay:{source}"


def _paths_fingerprint(
    paths: tuple[str | Path, ...],
    *,
    require_non_empty: bool = False,
) -> str:
    if require_non_empty and not paths:
        raise ValueError("At least one path is required")
    digest = hashlib.sha256()
    for source in paths:
        content = Path(source).read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()
