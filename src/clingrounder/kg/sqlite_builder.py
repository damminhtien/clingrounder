"""Build an immutable SQLite index from compiled knowledge-graph JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clingrounder.preprocessing.normalizer import NORMALIZATION_CONTRACT_VERSION
from clingrounder.utils.hashing import sha256_file
from clingrounder.utils.text import normalize_for_match

__all__ = [
    "KNOWLEDGE_GRAPH_SCHEMA_VERSION",
    "KnowledgeGraphIndexManifest",
    "build_knowledge_graph_index",
    "graph_input_fingerprint",
    "knowledge_graph_cache_path",
]

KNOWLEDGE_GRAPH_SCHEMA_VERSION = "sqlite-knowledge-graph-v2"


@dataclass(frozen=True)
class KnowledgeGraphIndexManifest:
    """Content and row counts for one published graph index."""

    index_path: str
    schema_version: str
    normalization_version: str
    input_fingerprint: str
    node_count: int
    alias_count: int
    edge_count: int
    evidence_count: int

    def to_json(self) -> dict[str, str | int]:
        return asdict(self)


def graph_input_fingerprint(
    nodes_path: str | Path,
    edges_path: str | Path,
    evidence_path: str | Path,
) -> str:
    """Hash all graph tables with domain separation."""

    digest = hashlib.sha256()
    for label, path in (
        (b"nodes", nodes_path),
        (b"edges", edges_path),
        (b"evidence", evidence_path),
    ):
        digest.update(label)
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def knowledge_graph_cache_path(
    cache_dir: str | Path,
    nodes_path: str | Path,
    edges_path: str | Path,
    evidence_path: str | Path,
    *,
    schema_version: str = KNOWLEDGE_GRAPH_SCHEMA_VERSION,
    normalization_version: str = NORMALIZATION_CONTRACT_VERSION,
) -> Path:
    """Derive a cache path from graph bytes and schema contracts."""

    digest = hashlib.sha256()
    digest.update(graph_input_fingerprint(nodes_path, edges_path, evidence_path).encode("ascii"))
    digest.update(schema_version.encode("utf-8"))
    digest.update(normalization_version.encode("utf-8"))
    return Path(cache_dir) / f"knowledge-graph-{digest.hexdigest()[:20]}.sqlite3"


def build_knowledge_graph_index(
    nodes_path: str | Path,
    edges_path: str | Path,
    evidence_path: str | Path,
    *,
    output_path: str | Path | None = None,
    cache_dir: str | Path = ".cache/clingrounder/knowledge-graph",
    normalization_version: str = NORMALIZATION_CONTRACT_VERSION,
) -> KnowledgeGraphIndexManifest:
    """Validate, index, and atomically publish a compiled graph."""

    fingerprint = graph_input_fingerprint(nodes_path, edges_path, evidence_path)
    target = (
        Path(output_path)
        if output_path is not None
        else knowledge_graph_cache_path(
            cache_dir,
            nodes_path,
            edges_path,
            evidence_path,
            normalization_version=normalization_version,
        )
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        connection = sqlite3.connect(temporary)
        try:
            _initialize_schema(connection)
            counts = _write_graph(
                connection,
                nodes_path=Path(nodes_path),
                edges_path=Path(edges_path),
                evidence_path=Path(evidence_path),
                input_fingerprint=fingerprint,
                normalization_version=normalization_version,
            )
        finally:
            connection.close()
        # SCALING: readers observe either the prior complete graph or this complete graph.
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return KnowledgeGraphIndexManifest(
        index_path=str(target),
        schema_version=KNOWLEDGE_GRAPH_SCHEMA_VERSION,
        normalization_version=normalization_version,
        input_fingerprint=fingerprint,
        node_count=counts[0],
        alias_count=counts[1],
        edge_count=counts[2],
        evidence_count=counts[3],
    )


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA foreign_keys = ON;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            label TEXT NOT NULL,
            normalized_label TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            code_system TEXT,
            code TEXT,
            aliases_json TEXT NOT NULL,
            terminology_versions_json TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL,
            document_count INTEGER NOT NULL,
            CHECK ((kind = 'CONCEPT' AND code_system IS NOT NULL AND code IS NOT NULL)
                OR (kind = 'TERM' AND code_system IS NULL AND code IS NULL))
        ) WITHOUT ROWID;
        CREATE UNIQUE INDEX nodes_code_idx
            ON nodes(code_system, code) WHERE code_system IS NOT NULL AND code IS NOT NULL;
        CREATE INDEX nodes_normalized_idx
            ON nodes(normalized_label, entity_type, code_system);
        CREATE TABLE node_aliases (
            node_id TEXT NOT NULL REFERENCES nodes(node_id),
            surface TEXT NOT NULL,
            normalized TEXT NOT NULL,
            toneless TEXT NOT NULL,
            PRIMARY KEY (node_id, normalized)
        ) WITHOUT ROWID;
        CREATE INDEX node_aliases_normalized_idx
            ON node_aliases(normalized, node_id);
        CREATE INDEX node_aliases_toneless_idx
            ON node_aliases(toneless, node_id);
        CREATE VIRTUAL TABLE nodes_fts USING fts5(
            node_id UNINDEXED,
            label,
            aliases,
            tokenize = 'unicode61 remove_diacritics 2'
        );
        CREATE TABLE edges (
            edge_id TEXT PRIMARY KEY,
            head_node_id TEXT NOT NULL REFERENCES nodes(node_id),
            tail_node_id TEXT NOT NULL REFERENCES nodes(node_id),
            relation_type TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            document_count INTEGER NOT NULL,
            confidence_mean REAL NOT NULL,
            confidence_min REAL NOT NULL,
            confidence_max REAL NOT NULL,
            sources_json TEXT NOT NULL,
            layers_json TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX edges_head_idx
            ON edges(head_node_id, relation_type, support_count DESC, tail_node_id);
        CREATE INDEX edges_tail_idx
            ON edges(tail_node_id, relation_type, support_count DESC, head_node_id);
        CREATE TABLE evidence (
            evidence_id TEXT PRIMARY KEY,
            edge_id TEXT NOT NULL REFERENCES edges(edge_id),
            source_record_id TEXT NOT NULL,
            source_record_kind TEXT NOT NULL,
            source TEXT NOT NULL,
            document_id TEXT,
            source_artifact_id TEXT,
            evidence_start INTEGER,
            evidence_end INTEGER,
            head_annotation_id TEXT,
            tail_annotation_id TEXT
        ) WITHOUT ROWID;
        CREATE INDEX evidence_edge_idx ON evidence(edge_id, evidence_id);
        """
    )


def _write_graph(
    connection: sqlite3.Connection,
    *,
    nodes_path: Path,
    edges_path: Path,
    evidence_path: Path,
    input_fingerprint: str,
    normalization_version: str,
) -> tuple[int, int, int, int]:
    node_count = 0
    alias_count = 0
    for line_number, raw in _jsonl_rows(nodes_path):
        node_id = _required_string(raw, "node_id", nodes_path, line_number)
        label = _required_string(raw, "label", nodes_path, line_number)
        aliases = _string_list(raw, "aliases", nodes_path, line_number)
        normalized = normalize_for_match(label)
        declared_normalized = _required_string(
            raw, "normalized_label", nodes_path, line_number
        )
        if normalized != declared_normalized:
            raise ValueError(f"{nodes_path}:{line_number}: normalized_label is stale")
        connection.execute(
            """
            INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                _required_string(raw, "kind", nodes_path, line_number),
                label,
                normalized,
                _required_string(raw, "entity_type", nodes_path, line_number),
                _optional_string(raw.get("code_system")),
                _optional_string(raw.get("code")),
                json.dumps(aliases, ensure_ascii=False, sort_keys=True),
                json.dumps(
                    _string_list(raw, "terminology_versions", nodes_path, line_number),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                json.dumps(
                    _string_list(raw, "sources", nodes_path, line_number),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                _non_negative_int(raw, "occurrence_count", nodes_path, line_number),
                _non_negative_int(raw, "document_count", nodes_path, line_number),
            ),
        )
        connection.execute(
            "INSERT INTO nodes_fts(node_id, label, aliases) VALUES (?, ?, ?)",
            (node_id, label, " ".join(aliases)),
        )
        # SCALING: materialize exact keys once at build time; runtime lookup avoids
        # decoding every node's aliases_json or scanning the full FTS candidate set.
        alias_surfaces: dict[str, str] = {}
        for surface in (label, *aliases):
            alias_normalized = normalize_for_match(surface)
            if alias_normalized:
                alias_surfaces.setdefault(alias_normalized, surface)
        for alias_normalized, surface in sorted(alias_surfaces.items()):
            connection.execute(
                "INSERT INTO node_aliases VALUES (?, ?, ?, ?)",
                (
                    node_id,
                    surface,
                    alias_normalized,
                    normalize_for_match(surface, strip_diacritics=True),
                ),
            )
            alias_count += 1
        node_count += 1

    edge_count = 0
    for line_number, raw in _jsonl_rows(edges_path):
        connection.execute(
            """
            INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _required_string(raw, "edge_id", edges_path, line_number),
                _required_string(raw, "head_node_id", edges_path, line_number),
                _required_string(raw, "tail_node_id", edges_path, line_number),
                _required_string(raw, "relation_type", edges_path, line_number),
                _positive_int(raw, "support_count", edges_path, line_number),
                _non_negative_int(raw, "document_count", edges_path, line_number),
                _probability(raw, "confidence_mean", edges_path, line_number),
                _probability(raw, "confidence_min", edges_path, line_number),
                _probability(raw, "confidence_max", edges_path, line_number),
                json.dumps(
                    _string_list(raw, "sources", edges_path, line_number),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                json.dumps(
                    _string_list(raw, "layers", edges_path, line_number),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        )
        edge_count += 1

    evidence_count = 0
    for line_number, raw in _jsonl_rows(evidence_path):
        span = raw.get("evidence_span")
        start: int | None = None
        end: int | None = None
        if span is not None:
            if (
                not isinstance(span, list)
                or len(span) != 2
                or isinstance(span[0], bool)
                or isinstance(span[1], bool)
            ):
                raise ValueError(f"{evidence_path}:{line_number}: invalid evidence_span")
            start, end = int(span[0]), int(span[1])
            if start < 0 or end <= start:
                raise ValueError(f"{evidence_path}:{line_number}: invalid evidence_span")
        connection.execute(
            """
            INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _required_string(raw, "evidence_id", evidence_path, line_number),
                _required_string(raw, "edge_id", evidence_path, line_number),
                _required_string(raw, "source_record_id", evidence_path, line_number),
                _required_string(raw, "source_record_kind", evidence_path, line_number),
                _required_string(raw, "source", evidence_path, line_number),
                _optional_string(raw.get("document_id")),
                _optional_string(raw.get("source_artifact_id")),
                start,
                end,
                _optional_string(raw.get("head_annotation_id")),
                _optional_string(raw.get("tail_annotation_id")),
            ),
        )
        evidence_count += 1

    metadata = {
        "schema_version": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
        "normalization_version": normalization_version,
        "input_fingerprint": input_fingerprint,
        "node_count": str(node_count),
        "alias_count": str(alias_count),
        "edge_count": str(edge_count),
        "evidence_count": str(evidence_count),
    }
    connection.executemany("INSERT INTO metadata VALUES (?, ?)", sorted(metadata.items()))
    connection.commit()
    return node_count, alias_count, edge_count, evidence_count


def _jsonl_rows(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Stream source rows so full graph size does not determine Python RSS."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield line_number, raw


def _required_string(
    raw: Mapping[str, Any], field: str, path: Path, line_number: int
) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: {field} must be a non-empty string")
    return value.strip()


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _string_list(
    raw: Mapping[str, Any], field: str, path: Path, line_number: int
) -> list[str]:
    value = raw.get(field, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{path}:{line_number}: {field} must be a string array")
    return [str(item) for item in value]


def _non_negative_int(
    raw: Mapping[str, Any], field: str, path: Path, line_number: int
) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}:{line_number}: {field} must be a non-negative integer")
    return value


def _positive_int(
    raw: Mapping[str, Any], field: str, path: Path, line_number: int
) -> int:
    value = _non_negative_int(raw, field, path, line_number)
    if value < 1:
        raise ValueError(f"{path}:{line_number}: {field} must be positive")
    return value


def _probability(
    raw: Mapping[str, Any], field: str, path: Path, line_number: int
) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path}:{line_number}: {field} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path}:{line_number}: {field} must be in [0, 1]")
    return result
