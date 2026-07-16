"""Build a versioned SQLite FTS5 index from canonical terminology JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.preprocessing.normalizer import NORMALIZATION_CONTRACT_VERSION
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "TERMINOLOGY_INDEX_SCHEMA_VERSION",
    "TerminologyIndexManifest",
    "build_terminology_index",
    "source_fingerprint",
    "terminology_cache_path",
]

TERMINOLOGY_INDEX_SCHEMA_VERSION = "sqlite-fts5-v1"


@dataclass(frozen=True)
class TerminologyIndexManifest:
    """Reproducibility metadata emitted for one derived index."""

    index_path: str
    schema_version: str
    normalization_version: str
    source_fingerprint: str
    concept_count: int
    alias_count: int

    def to_json(self) -> dict[str, str | int]:
        """Return a JSON-serializable manifest payload."""

        return asdict(self)


def source_fingerprint(source_paths: tuple[str | Path, ...]) -> str:
    """Hash source bytes in declared merge order."""

    if not source_paths:
        raise ValueError("At least one terminology source is required")
    digest = hashlib.sha256()
    for source in source_paths:
        path = Path(source)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def terminology_cache_path(
    cache_dir: str | Path,
    source_paths: tuple[str | Path, ...],
    *,
    normalization_version: str = NORMALIZATION_CONTRACT_VERSION,
    schema_version: str = TERMINOLOGY_INDEX_SCHEMA_VERSION,
) -> Path:
    """Derive a content-addressed cache path for the source and schema contract."""

    digest = hashlib.sha256()
    digest.update(source_fingerprint(source_paths).encode("ascii"))
    digest.update(schema_version.encode("utf-8"))
    digest.update(normalization_version.encode("utf-8"))
    return Path(cache_dir) / f"terminology-{digest.hexdigest()[:20]}.sqlite3"


def build_terminology_index(
    source_paths: tuple[str | Path, ...],
    *,
    output_path: str | Path | None = None,
    cache_dir: str | Path = ".cache/medical-kg/terminology",
    normalization_version: str = NORMALIZATION_CONTRACT_VERSION,
) -> TerminologyIndexManifest:
    """Build into a sibling temporary file and atomically publish on success."""

    fingerprint = source_fingerprint(source_paths)
    target = (
        Path(output_path)
        if output_path is not None
        else terminology_cache_path(
            cache_dir,
            source_paths,
            normalization_version=normalization_version,
        )
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    entries = _load_entries(source_paths)
    try:
        connection = sqlite3.connect(temporary)
        try:
            _initialize_schema(connection)
            alias_count = _write_entries(
                connection,
                entries,
                source_paths=source_paths,
                source_fingerprint_value=fingerprint,
                normalization_version=normalization_version,
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
        source_fingerprint=fingerprint,
        concept_count=len(entries),
        alias_count=alias_count,
    )


def _load_entries(source_paths: tuple[str | Path, ...]) -> list[ConceptEntry]:
    entries: list[ConceptEntry] = []
    for source_path in source_paths:
        entries.extend(DictionaryStore.load_entries_jsonl(source_path))
    by_concept_id = {entry.concept_id: entry for entry in entries}
    return list(by_concept_id.values())


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
    source_fingerprint_value: str,
    normalization_version: str,
) -> int:
    metadata = {
        "schema_version": TERMINOLOGY_INDEX_SCHEMA_VERSION,
        "normalization_version": normalization_version,
        "source_fingerprint": source_fingerprint_value,
        "source_paths": json.dumps([str(Path(path)) for path in source_paths]),
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
            for surface, normalized, toneless, source in _alias_rows(entry)
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
            [(surface, normalized, toneless, concept_id) for surface, normalized, toneless, _, concept_id in alias_rows],
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


def _alias_rows(entry: ConceptEntry) -> list[tuple[str, str, str, str]]:
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
        output.append(
            (
                surface,
                normalize_for_match(surface),
                normalize_for_match(surface, strip_diacritics=True),
                source,
            )
        )
    return output
