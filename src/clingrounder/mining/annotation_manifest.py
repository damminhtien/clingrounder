"""Disk-backed, deterministic materialization of mined annotation manifests.

Large sources can emit millions of structured spans. Keeping every proposal in a
Python tuple merely to sort by ID makes source acquisition scalable but labeling
non-scalable. This module uses a disposable SQLite index for bounded-memory sorting
and exact duplicate detection; JSONL remains the canonical portable artifact.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.mining.io import write_jsonl
from clingrounder.mining.records import AnnotationProposal

__all__ = ["AnnotationManifestResult", "materialize_annotation_manifest"]


@dataclass(frozen=True)
class AnnotationManifestResult:
    """Counts and fingerprint returned after one atomic annotation write."""

    path: str
    annotation_count: int
    duplicate_count: int
    sha256: str


def materialize_annotation_manifest(
    path: str | Path,
    annotations: Iterable[AnnotationProposal],
) -> AnnotationManifestResult:
    """Sort and deduplicate a proposal stream without retaining it in memory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.index-",
        dir=target.parent,
    ) as temporary_dir:
        database_path = Path(temporary_dir) / "annotations.sqlite3"
        with sqlite3.connect(database_path) as connection:
            _prepare_database(connection)
            annotation_count = 0
            duplicate_count = 0
            for annotation in annotations:
                if _insert_annotation(connection, annotation):
                    annotation_count += 1
                else:
                    duplicate_count += 1
            connection.commit()
            sha256 = write_jsonl(target, _iter_payloads(connection))
    return AnnotationManifestResult(
        path=str(target),
        annotation_count=annotation_count,
        duplicate_count=duplicate_count,
        sha256=sha256,
    )


def _prepare_database(connection: sqlite3.Connection) -> None:
    # SCALING: this scratch index is rebuilt from immutable proposals; the final
    # JSONL write is atomic, so SQLite journaling would only duplicate IO.
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE annotations ("
        "annotation_id TEXT PRIMARY KEY, payload TEXT NOT NULL"
        ") WITHOUT ROWID"
    )


def _insert_annotation(
    connection: sqlite3.Connection,
    annotation: AnnotationProposal,
) -> bool:
    encoded = _encode(annotation.to_dict())
    try:
        connection.execute(
            "INSERT INTO annotations(annotation_id, payload) VALUES (?, ?)",
            (annotation.annotation_id, encoded),
        )
        return True
    except sqlite3.IntegrityError:
        row = connection.execute(
            "SELECT payload FROM annotations WHERE annotation_id = ?",
            (annotation.annotation_id,),
        ).fetchone()
        if row is None:  # pragma: no cover - SQLite primary-key behavior
            raise RuntimeError("Annotation disappeared during manifest materialization")
        if str(row[0]) != encoded:
            # INVARIANT: one annotation ID cannot describe two medical labels.
            raise ValueError(f"Conflicting annotation ID {annotation.annotation_id!r}")
        return False


def _iter_payloads(connection: sqlite3.Connection) -> Iterator[Mapping[str, Any]]:
    cursor = connection.execute(
        "SELECT payload FROM annotations ORDER BY annotation_id"
    )
    for (payload,) in cursor:
        value = json.loads(str(payload))
        if not isinstance(value, dict):  # pragma: no cover - encoded internally
            raise RuntimeError("Stored annotation payload is not an object")
        yield value


def _encode(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
