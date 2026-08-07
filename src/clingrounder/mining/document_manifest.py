"""Disk-backed, deterministic materialization of mined document manifests."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.mining.io import write_jsonl
from clingrounder.mining.records import MinedDocument

__all__ = ["DocumentManifestResult", "materialize_document_manifest"]

_ORIGIN_METADATA_KEYS = frozenset(
    {
        "archive_member",
        "dailymed_source_version",
        "published_date",
        "source_archive_members",
        "source_artifact_ids",
        "source_published_dates",
        "source_versions",
    }
)


@dataclass(frozen=True)
class DocumentManifestResult:
    """Counts and fingerprint returned after one atomic manifest write."""

    path: str
    document_count: int
    duplicate_count: int
    sha256: str


def materialize_document_manifest(
    path: str | Path,
    documents: Iterable[MinedDocument],
) -> DocumentManifestResult:
    """Deduplicate an arbitrary document stream through a temporary SQLite index.

    SQLite bounds Python memory and provides stable primary-key ordering. The database is a
    scratch index only; JSONL remains the portable canonical stage artifact.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.index-",
        dir=target.parent,
    ) as temporary_dir:
        database_path = Path(temporary_dir) / "documents.sqlite3"
        with sqlite3.connect(database_path) as connection:
            _prepare_database(connection)
            document_count = 0
            duplicate_count = 0
            for document in documents:
                inserted = _upsert_document(connection, document)
                if inserted:
                    document_count += 1
                else:
                    duplicate_count += 1
            connection.commit()
            sha256 = write_jsonl(target, _iter_payloads(connection))
    return DocumentManifestResult(
        path=str(target),
        document_count=document_count,
        duplicate_count=duplicate_count,
        sha256=sha256,
    )


def _prepare_database(connection: sqlite3.Connection) -> None:
    # SCALING: this database is disposable and rebuilt from immutable source records. Disabling
    # journaling avoids duplicate writes while the final JSONL still uses atomic replacement.
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE documents ("
        "document_id TEXT PRIMARY KEY, payload TEXT NOT NULL, signature TEXT NOT NULL"
        ") WITHOUT ROWID"
    )


def _upsert_document(connection: sqlite3.Connection, document: MinedDocument) -> bool:
    payload = document.to_dict()
    encoded = _encode(payload)
    signature = _encode(_content_signature(payload))
    try:
        connection.execute(
            "INSERT INTO documents(document_id, payload, signature) VALUES (?, ?, ?)",
            (document.document_id, encoded, signature),
        )
        return True
    except sqlite3.IntegrityError:
        row = connection.execute(
            "SELECT payload, signature FROM documents WHERE document_id = ?",
            (document.document_id,),
        ).fetchone()
        if row is None:  # pragma: no cover - SQLite primary-key behavior
            raise RuntimeError("Document disappeared during manifest materialization")
        previous_payload = _object(json.loads(str(row[0])))
        if str(row[1]) != signature:
            raise ValueError(f"Conflicting document ID {document.document_id!r}")
        merged = _merge_origins(previous_payload, payload)
        if merged != previous_payload:
            connection.execute(
                "UPDATE documents SET payload = ? WHERE document_id = ?",
                (_encode(merged), document.document_id),
            )
        return False


def _iter_payloads(connection: sqlite3.Connection) -> Iterator[Mapping[str, Any]]:
    cursor = connection.execute("SELECT payload FROM documents ORDER BY document_id")
    for (payload,) in cursor:
        yield _object(json.loads(str(payload)))


def _content_signature(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("source_artifact_id", None)
    metadata = _object(result.get("metadata", {}))
    result["metadata"] = {
        key: value for key, value in metadata.items() if key not in _ORIGIN_METADATA_KEYS
    }
    return result


def _merge_origins(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    previous_metadata = _object(previous.get("metadata", {}))
    current_metadata = _object(current.get("metadata", {}))
    previous_key = (
        str(previous["source_artifact_id"]),
        str(previous_metadata.get("archive_member", "")),
    )
    current_key = (
        str(current["source_artifact_id"]),
        str(current_metadata.get("archive_member", "")),
    )
    primary = dict(previous if previous_key <= current_key else current)
    primary_metadata = dict(
        previous_metadata if previous_key <= current_key else current_metadata
    )
    artifact_ids = _origin_values(previous, "source_artifact_ids", "source_artifact_id")
    artifact_ids.update(_origin_values(current, "source_artifact_ids", "source_artifact_id"))
    _store_origin_values(primary_metadata, "source_artifact_ids", artifact_ids)

    archive_members = _metadata_origin_values(previous_metadata, "source_archive_members")
    archive_members.update(_metadata_origin_values(current_metadata, "source_archive_members"))
    for payload, metadata in (
        (previous, previous_metadata),
        (current, current_metadata),
    ):
        member = str(metadata.get("archive_member", ""))
        if member:
            archive_members.add(f"{payload['source_artifact_id']}:{member}")
    _store_origin_values(primary_metadata, "source_archive_members", archive_members)

    source_versions = _metadata_origin_values(previous_metadata, "source_versions")
    source_versions.update(_metadata_origin_values(current_metadata, "source_versions"))
    for metadata in (previous_metadata, current_metadata):
        value = str(metadata.get("dailymed_source_version", ""))
        if value:
            source_versions.add(value)
    _store_origin_values(primary_metadata, "source_versions", source_versions)

    published_dates = _metadata_origin_values(previous_metadata, "source_published_dates")
    published_dates.update(_metadata_origin_values(current_metadata, "source_published_dates"))
    for metadata in (previous_metadata, current_metadata):
        value = str(metadata.get("published_date", ""))
        if value:
            published_dates.add(value)
    _store_origin_values(primary_metadata, "source_published_dates", published_dates)
    primary["metadata"] = primary_metadata
    return primary


def _origin_values(
    payload: Mapping[str, Any],
    metadata_key: str,
    payload_key: str,
) -> set[str]:
    metadata = _object(payload.get("metadata", {}))
    values = _metadata_origin_values(metadata, metadata_key)
    values.add(str(payload[payload_key]))
    return values


def _metadata_origin_values(metadata: Mapping[str, Any], key: str) -> set[str]:
    raw = metadata.get(key)
    if raw is None:
        return set()
    try:
        values = json.loads(str(raw))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid document origin metadata {key!r}") from error
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"Document origin metadata {key!r} must be a JSON string list")
    return set(values)


def _store_origin_values(metadata: dict[str, Any], key: str, values: set[str]) -> None:
    if len(values) > 1:
        metadata[key] = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":"))
    else:
        metadata.pop(key, None)


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Document manifest payload must be a JSON object")
    return value


def _encode(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
