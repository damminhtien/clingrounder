"""Read-only, thread-local queries over a compiled SQLite knowledge graph."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

from clingrounder.kg.knowledge_schema import (
    KnowledgeEdge,
    KnowledgeEvidence,
    KnowledgeNeighbor,
    KnowledgeNode,
    KnowledgeNodeKind,
)
from clingrounder.kg.sqlite_builder import (
    KNOWLEDGE_GRAPH_SCHEMA_VERSION,
    graph_input_fingerprint,
)
from clingrounder.preprocessing.normalizer import NORMALIZATION_CONTRACT_VERSION
from clingrounder.utils.text import normalize_for_match

__all__ = ["KnowledgeNeighbor", "SQLiteKnowledgeGraphRepository"]

_NODE_COLUMNS = (
    "n.node_id, n.kind, n.label, n.normalized_label, n.entity_type, "
    "n.code_system, n.code, n.aliases_json, n.terminology_versions_json, "
    "n.sources_json, n.occurrence_count, n.document_count"
)
_EDGE_COLUMNS = (
    "e.edge_id, e.head_node_id, e.tail_node_id, e.relation_type, "
    "e.support_count, e.document_count, e.confidence_mean, e.confidence_min, "
    "e.confidence_max, e.sources_json, e.layers_json"
)

class SQLiteKnowledgeGraphRepository:
    """Query a graph index without loading nodes or evidence into Python memory."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        expected_nodes_path: str | Path | None = None,
        expected_edges_path: str | Path | None = None,
        expected_evidence_path: str | Path | None = None,
        expected_normalization_version: str = NORMALIZATION_CONTRACT_VERSION,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Knowledge graph index does not exist: {self.index_path}")
        self._local = threading.local()
        self._connections: dict[int, sqlite3.Connection] = {}
        self._connections_lock = threading.RLock()
        self._closed = False
        self.metadata = self._load_metadata()
        self._validate_metadata(
            expected_nodes_path,
            expected_edges_path,
            expected_evidence_path,
            expected_normalization_version,
        )

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        row = self._connection().execute(
            f"SELECT {_NODE_COLUMNS} FROM nodes n WHERE n.node_id = ?",
            (node_id,),
        ).fetchone()
        return None if row is None else _node_from_row(row)

    def get_by_code(self, code_system: str, code: str) -> KnowledgeNode | None:
        row = self._connection().execute(
            f"SELECT {_NODE_COLUMNS} FROM nodes n WHERE n.code_system = ? AND n.code = ?",
            (code_system, code),
        ).fetchone()
        return None if row is None else _node_from_row(row)

    def search_nodes(
        self,
        query: str,
        *,
        entity_type: str | None = None,
        code_system: str | None = None,
        limit: int = 20,
        exact_only: bool = False,
    ) -> list[KnowledgeNode]:
        """Search exact/toneless aliases, optionally skipping the FTS fallback."""

        _validate_limit(limit)
        normalized = normalize_for_match(query)
        conditions, parameters = _node_filters(entity_type, code_system)
        exact_rows = self._connection().execute(
            f"""
            SELECT {_NODE_COLUMNS}
            FROM node_aliases a
            JOIN nodes n ON n.node_id = a.node_id
            WHERE a.normalized = ? {conditions}
            ORDER BY n.occurrence_count DESC, n.node_id
            LIMIT ?
            """,
            (normalized, *parameters, limit),
        ).fetchall()
        output = [_node_from_row(row) for row in exact_rows]
        if len(output) >= limit or len(normalized) < 2:
            return output
        seen = {node.node_id for node in output}
        toneless = normalize_for_match(query, strip_diacritics=True)
        toneless_rows = self._connection().execute(
            f"""
            SELECT {_NODE_COLUMNS}
            FROM node_aliases a
            JOIN nodes n ON n.node_id = a.node_id
            WHERE a.toneless = ? {conditions}
            ORDER BY n.occurrence_count DESC, n.node_id
            LIMIT ?
            """,
            (toneless, *parameters, limit * 4),
        )
        for row in toneless_rows:
            node = _node_from_row(row)
            if node.node_id in seen:
                continue
            output.append(node)
            seen.add(node.node_id)
            if len(output) >= limit:
                return output
        if exact_only:
            return output
        match_query = f'"{normalized.replace(chr(34), chr(34) * 2)}"'
        fts_rows = self._connection().execute(
            f"""
            SELECT {_NODE_COLUMNS}, bm25(nodes_fts) AS lexical_rank
            FROM nodes_fts
            JOIN nodes n ON n.node_id = nodes_fts.node_id
            WHERE nodes_fts MATCH ? {conditions}
            ORDER BY lexical_rank, n.occurrence_count DESC, n.node_id
            LIMIT ?
            """,
            (match_query, *parameters, limit * 4),
        )
        for row in fts_rows:
            node = _node_from_row(row)
            if node.node_id in seen:
                continue
            output.append(node)
            seen.add(node.node_id)
            if len(output) >= limit:
                break
        return output

    def neighbors(
        self,
        node_id: str,
        *,
        direction: str = "outgoing",
        relation_types: Sequence[str] = (),
        min_support: int = 1,
        limit: int = 100,
    ) -> list[KnowledgeNeighbor]:
        """Return deterministically ranked adjacent nodes with optional edge filters."""

        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("direction must be outgoing, incoming, or both")
        if min_support < 1:
            raise ValueError("min_support must be positive")
        _validate_limit(limit)
        output: list[KnowledgeNeighbor] = []
        if direction in {"outgoing", "both"}:
            output.extend(
                self._neighbors_one_direction(
                    node_id,
                    outgoing=True,
                    relation_types=relation_types,
                    min_support=min_support,
                    limit=limit,
                )
            )
        if direction in {"incoming", "both"} and len(output) < limit:
            output.extend(
                self._neighbors_one_direction(
                    node_id,
                    outgoing=False,
                    relation_types=relation_types,
                    min_support=min_support,
                    limit=limit - len(output),
                )
            )
        return output[:limit]

    def ancestors(self, node_id: str, *, max_depth: int = 20) -> list[tuple[KnowledgeNode, int]]:
        """Traverse `IS_A` edges with cycle protection in SQLite."""

        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        rows = self._connection().execute(
            f"""
            WITH RECURSIVE paths(node_id, distance, visited) AS (
                SELECT e.tail_node_id, 1, ',' || e.head_node_id || ',' || e.tail_node_id || ','
                FROM edges e
                WHERE e.head_node_id = ? AND e.relation_type = 'IS_A'
                UNION ALL
                SELECT e.tail_node_id, p.distance + 1, p.visited || e.tail_node_id || ','
                FROM paths p
                JOIN edges e ON e.head_node_id = p.node_id
                WHERE e.relation_type = 'IS_A'
                  AND p.distance < ?
                  AND instr(p.visited, ',' || e.tail_node_id || ',') = 0
            )
            SELECT {_NODE_COLUMNS}, MIN(paths.distance) AS distance
            FROM paths
            JOIN nodes n ON n.node_id = paths.node_id
            GROUP BY n.node_id
            ORDER BY distance, n.node_id
            """,
            (node_id, max_depth),
        )
        return [(_node_from_row(row), int(row[12])) for row in rows]

    def evidence(self, edge_id: str, *, limit: int = 100) -> list[KnowledgeEvidence]:
        _validate_limit(limit)
        rows = self._connection().execute(
            """
            SELECT evidence_id, edge_id, source_record_id, source_record_kind, source,
                   document_id, source_artifact_id, evidence_start, evidence_end,
                   head_annotation_id, tail_annotation_id
            FROM evidence
            WHERE edge_id = ?
            ORDER BY evidence_id
            LIMIT ?
            """,
            (edge_id, limit),
        )
        return [_evidence_from_row(row) for row in rows]

    def close(self) -> None:
        with self._connections_lock:
            if self._closed:
                return
            self._closed = True
            connections = tuple(self._connections.values())
            self._connections.clear()
        for connection in connections:
            connection.close()
        if hasattr(self._local, "connection"):
            del self._local.connection

    def _neighbors_one_direction(
        self,
        node_id: str,
        *,
        outgoing: bool,
        relation_types: Sequence[str],
        min_support: int,
        limit: int,
    ) -> list[KnowledgeNeighbor]:
        endpoint = "e.head_node_id" if outgoing else "e.tail_node_id"
        adjacent = "e.tail_node_id" if outgoing else "e.head_node_id"
        relation_sql = ""
        parameters: list[object] = [node_id, min_support]
        if relation_types:
            relation_sql = f" AND e.relation_type IN ({','.join('?' for _ in relation_types)})"
            parameters.extend(relation_types)
        parameters.append(limit)
        rows = self._connection().execute(
            f"""
            SELECT {_EDGE_COLUMNS}, {_NODE_COLUMNS}
            FROM edges e
            JOIN nodes n ON n.node_id = {adjacent}
            WHERE {endpoint} = ? AND e.support_count >= ? {relation_sql}
            ORDER BY e.support_count DESC, e.confidence_mean DESC,
                     e.relation_type, n.node_id
            LIMIT ?
            """,
            parameters,
        )
        return [
            KnowledgeNeighbor(
                edge=_edge_from_row(row),
                node=_node_from_row(row, offset=11),
                direction="outgoing" if outgoing else "incoming",
            )
            for row in rows
        ]

    def _connection(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("SQLite knowledge graph repository is closed")
        connection = getattr(self._local, "connection", None)
        if connection is None:
            # SCALING: immutable thread-local readers avoid a global connection lock.
            uri = f"file:{quote(str(self.index_path))}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            self._local.connection = connection
            with self._connections_lock:
                if self._closed:
                    connection.close()
                    del self._local.connection
                    raise RuntimeError("SQLite knowledge graph repository is closed")
                self._connections[threading.get_ident()] = connection
        return connection

    def _load_metadata(self) -> dict[str, str]:
        uri = f"file:{quote(str(self.index_path))}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        try:
            return {
                str(key): str(value)
                for key, value in connection.execute("SELECT key, value FROM metadata")
            }
        except sqlite3.DatabaseError as error:
            raise ValueError(f"Invalid knowledge graph index: {self.index_path}") from error
        finally:
            connection.close()

    def _validate_metadata(
        self,
        expected_nodes_path: str | Path | None,
        expected_edges_path: str | Path | None,
        expected_evidence_path: str | Path | None,
        expected_normalization_version: str,
    ) -> None:
        if self.metadata.get("schema_version") != KNOWLEDGE_GRAPH_SCHEMA_VERSION:
            raise ValueError("Knowledge graph index schema version is stale")
        if self.metadata.get("normalization_version") != expected_normalization_version:
            raise ValueError("Knowledge graph normalization contract is stale")
        expected_paths = (
            expected_nodes_path,
            expected_edges_path,
            expected_evidence_path,
        )
        if any(path is not None for path in expected_paths):
            if any(path is None for path in expected_paths):
                raise ValueError("All expected graph table paths are required for validation")
            assert expected_nodes_path is not None
            assert expected_edges_path is not None
            assert expected_evidence_path is not None
            current = graph_input_fingerprint(
                expected_nodes_path,
                expected_edges_path,
                expected_evidence_path,
            )
            if self.metadata.get("input_fingerprint") != current:
                raise ValueError("Knowledge graph input fingerprint is stale")


def _node_filters(entity_type: str | None, code_system: str | None) -> tuple[str, list[str]]:
    clauses: list[str] = []
    parameters: list[str] = []
    if entity_type is not None:
        clauses.append("n.entity_type = ?")
        parameters.append(entity_type)
    if code_system is not None:
        clauses.append("n.code_system = ?")
        parameters.append(code_system)
    return (" AND " + " AND ".join(clauses) if clauses else ""), parameters


def _node_from_row(row: sqlite3.Row, *, offset: int = 0) -> KnowledgeNode:
    return KnowledgeNode(
        node_id=str(row[offset]),
        kind=KnowledgeNodeKind(str(row[offset + 1])),
        label=str(row[offset + 2]),
        normalized_label=str(row[offset + 3]),
        entity_type=str(row[offset + 4]),
        code_system=None if row[offset + 5] is None else str(row[offset + 5]),
        code=None if row[offset + 6] is None else str(row[offset + 6]),
        aliases=tuple(json.loads(str(row[offset + 7]))),
        terminology_versions=tuple(json.loads(str(row[offset + 8]))),
        sources=tuple(json.loads(str(row[offset + 9]))),
        occurrence_count=int(row[offset + 10]),
        document_count=int(row[offset + 11]),
    )


def _edge_from_row(row: sqlite3.Row) -> KnowledgeEdge:
    return KnowledgeEdge(
        edge_id=str(row[0]),
        head_node_id=str(row[1]),
        tail_node_id=str(row[2]),
        relation_type=str(row[3]),
        support_count=int(row[4]),
        document_count=int(row[5]),
        confidence_mean=float(row[6]),
        confidence_min=float(row[7]),
        confidence_max=float(row[8]),
        sources=tuple(json.loads(str(row[9]))),
        layers=tuple(json.loads(str(row[10]))),
    )


def _evidence_from_row(row: sqlite3.Row) -> KnowledgeEvidence:
    span = None
    if row[7] is not None and row[8] is not None:
        span = (int(row[7]), int(row[8]))
    return KnowledgeEvidence(
        evidence_id=str(row[0]),
        edge_id=str(row[1]),
        source_record_id=str(row[2]),
        source_record_kind=str(row[3]),
        source=str(row[4]),
        document_id=None if row[5] is None else str(row[5]),
        source_artifact_id=None if row[6] is None else str(row[6]),
        evidence_span=span,
        head_annotation_id=None if row[9] is None else str(row[9]),
        tail_annotation_id=None if row[10] is None else str(row[10]),
    )


def _validate_limit(limit: int) -> None:
    if limit < 1:
        raise ValueError("limit must be positive")
