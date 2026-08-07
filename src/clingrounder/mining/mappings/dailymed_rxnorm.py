"""Compile the official DailyMed SPL-to-RxNorm mapping into JSONL and SQLite."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import sqlite3
import tempfile
import threading
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, BinaryIO

from clingrounder.mining.io import write_json, write_jsonl
from clingrounder.mining.records import SourceArtifact
from clingrounder.preprocessing.normalizer import NORMALIZATION_CONTRACT_VERSION
from clingrounder.utils.text import normalize_for_match

__all__ = [
    "DailyMedRxNormConcept",
    "DailyMedRxNormMappingRepository",
    "audit_dailymed_rxnorm_mapping",
    "compile_dailymed_rxnorm_mapping",
]

_EXPECTED_MEMBER = "rxnorm_mappings.txt"
_EXPECTED_FIELDS = ("SETID", "SPL_VERSION", "RXCUI", "RXSTRING", "RXTTY")
_SUPPORTED_TTYS = frozenset({"BPCK", "GPCK", "PSN", "SBD", "SCD", "SY"})
_INDEX_SCHEMA_VERSION = "dailymed-rxnorm-mapping.v2"
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_MAX_MEMBER_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class DailyMedRxNormConcept:
    """One RxCUI linked to a versioned SPL with source aliases and TTYs."""

    rxcui: str
    rxstrings: tuple[str, ...]
    rxttys: tuple[str, ...]


class DailyMedRxNormMappingRepository:
    """Thread-local read-only lookup over the compiled full mapping database."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_source_sha256: str | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._local = threading.local()
        metadata = self.metadata
        if metadata.get("schema_version") != _INDEX_SCHEMA_VERSION:
            raise ValueError(f"Unsupported DailyMed mapping index: {self.path}")
        if (
            expected_source_sha256 is not None
            and metadata.get("source_sha256") != expected_source_sha256
        ):
            raise ValueError("DailyMed mapping source fingerprint changed")

    @property
    def metadata(self) -> dict[str, str]:
        connection = self._connection()
        return {
            str(key): str(value)
            for key, value in connection.execute(
                "SELECT key, value FROM metadata ORDER BY key"
            )
        }

    def lookup(self, set_id: str, spl_version: str | int) -> tuple[DailyMedRxNormConcept, ...]:
        """Return all source mappings for one exact SPL version."""

        rows = self._connection().execute(
            """
            SELECT rxcui, rxstring, rxtty
            FROM mapping_rows
            WHERE set_id = ? AND spl_version = ?
            ORDER BY rxcui, rxstring, rxtty
            """,
            (set_id.strip().lower(), str(spl_version).strip()),
        )
        return tuple(_concepts_from_rows(rows))

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            del self._local.connection

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            uri = f"file:{self.path.as_posix()}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            connection.execute("PRAGMA query_only=ON")
            self._local.connection = connection
        return connection


def compile_dailymed_rxnorm_mapping(
    artifact: SourceArtifact,
    stream: BinaryIO,
    *,
    output_path: str | Path,
    index_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Deduplicate one checksum-pinned mapping release and atomically index it."""

    if artifact.source_id != "dailymed_rxnorm_mappings":
        raise ValueError("DailyMed mapping compiler received the wrong source")
    if artifact.object.byte_size > _MAX_ARCHIVE_BYTES:
        raise ValueError("DailyMed mapping archive exceeds the compressed size limit")
    payload = stream.read(_MAX_ARCHIVE_BYTES + 1)
    if len(payload) > _MAX_ARCHIVE_BYTES:
        raise ValueError("DailyMed mapping archive exceeds the compressed size limit")

    output = Path(output_path)
    index = Path(index_path)
    report_target = Path(report_path)
    for target in (output, index, report_target):
        target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{index.name}.", suffix=".sqlite3", dir=index.parent
    )
    os.close(descriptor)
    temporary_index = Path(temporary_name)
    temporary_index.unlink()
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary_index)
        _create_schema(connection)
        input_count, tty_counts = _load_archive(connection, payload)
        _finalize_index(connection, artifact, input_count)
        grouped_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT 1 FROM mapping_rows GROUP BY set_id, spl_version, rxcui
                )
                """
            ).fetchone()[0]
        )
        unique_row_count = int(
            connection.execute("SELECT COUNT(*) FROM mapping_rows").fetchone()[0]
        )
        set_version_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM (SELECT 1 FROM mapping_rows GROUP BY set_id, spl_version)"
            ).fetchone()[0]
        )
        rxcui_count = int(
            connection.execute("SELECT COUNT(DISTINCT rxcui) FROM mapping_rows").fetchone()[0]
        )
        output_sha256 = write_jsonl(
            output,
            _mapping_records(connection, artifact),
        )
        connection.close()
        connection = None
        # SCALING: readers see either the prior complete index or this complete index.
        os.replace(temporary_index, index)
        index_sha256 = _sha256_file(index)
        report: dict[str, Any] = {
            "schema_version": "dailymed-rxnorm-compilation-report.v1",
            "source_artifact_id": artifact.artifact_id,
            "source_sha256": artifact.object.sha256,
            "source_version": artifact.source_version,
            "input_row_count": input_count,
            "unique_source_row_count": unique_row_count,
            "duplicate_source_row_count": input_count - unique_row_count,
            "mapping_count": grouped_count,
            "set_version_count": set_version_count,
            "unique_rxcui_count": rxcui_count,
            "rxtty_counts": dict(sorted(tty_counts.items())),
            "output": str(output),
            "output_sha256": output_sha256,
            "index": str(index),
            "index_sha256": index_sha256,
        }
        write_json(report_target, report)
        return report
    except BaseException:
        if connection is not None:
            connection.close()
        temporary_index.unlink(missing_ok=True)
        raise


def audit_dailymed_rxnorm_mapping(
    mapping_index_path: str | Path,
    terminology_index_path: str | Path,
    *,
    proposals_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Find official DailyMed aliases safe to review against a pinned RxNorm index."""

    mapping_path = Path(mapping_index_path).resolve()
    terminology_path = Path(terminology_index_path).resolve()
    connection = sqlite3.connect(f"file:{mapping_path.as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("ATTACH DATABASE ? AS terminology", (str(terminology_path),))
        mapping_metadata = _metadata(connection, schema="main")
        terminology_metadata = _metadata(connection, schema="terminology")
        if mapping_metadata.get("schema_version") != _INDEX_SCHEMA_VERSION:
            raise ValueError("DailyMed mapping index schema is stale")
        if (
            mapping_metadata.get("normalization_version")
            != terminology_metadata.get("normalization_version")
        ):
            raise ValueError("DailyMed and terminology normalization contracts differ")
        connection.execute("PRAGMA query_only=ON")
        counts = _mapping_audit_counts(connection)
        proposals_sha256 = write_jsonl(
            proposals_path,
            _missing_alias_proposals(connection, mapping_metadata),
        )
        report: dict[str, Any] = {
            "schema_version": "dailymed-rxnorm-audit-report.v1",
            **counts,
            "absent_code_samples": _absent_code_samples(connection, limit=20),
            "mapping_index": str(mapping_path),
            "mapping_source_sha256": mapping_metadata.get("source_sha256", ""),
            "terminology_index": str(terminology_path),
            "terminology_source_fingerprint": terminology_metadata.get(
                "source_fingerprint", ""
            ),
            "proposals": str(proposals_path),
            "proposals_sha256": proposals_sha256,
            "promotion_policy": "review_required_no_automatic_dictionary_merge",
        }
        write_json(report_path, report)
        return report
    finally:
        connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE mapping_rows (
            set_id TEXT NOT NULL,
            spl_version TEXT NOT NULL,
            rxcui TEXT NOT NULL,
            rxstring TEXT NOT NULL,
            normalized TEXT NOT NULL,
            rxtty TEXT NOT NULL,
            PRIMARY KEY (set_id, spl_version, rxcui, rxstring, rxtty)
        ) WITHOUT ROWID;
        """
    )


def _metadata(connection: sqlite3.Connection, *, schema: str) -> dict[str, str]:
    if schema not in {"main", "terminology"}:
        raise ValueError(f"Unsupported SQLite schema {schema!r}")
    return {
        str(key): str(value)
        for key, value in connection.execute(
            f"SELECT key, value FROM {schema}.metadata ORDER BY key"
        )
    }


def _mapping_audit_counts(connection: sqlite3.Connection) -> dict[str, int]:
    row = connection.execute(
        """
        WITH mapping_codes AS (
            SELECT DISTINCT rxcui FROM mapping_rows
        ),
        known_codes AS (
            SELECT DISTINCT code
            FROM terminology.concepts
            WHERE code_system = 'RxNorm' AND code IS NOT NULL
        ),
        mapping_aliases AS (
            SELECT DISTINCT rxcui, normalized FROM mapping_rows
        ),
        known_aliases AS (
            SELECT DISTINCT c.code AS rxcui, a.normalized
            FROM terminology.aliases a
            JOIN terminology.concepts c ON c.concept_id = a.concept_id
            WHERE c.code_system = 'RxNorm' AND c.code IS NOT NULL
        )
        SELECT
            (SELECT COUNT(*) FROM mapping_codes),
            (SELECT COUNT(*) FROM mapping_codes m JOIN known_codes k ON k.code = m.rxcui),
            (SELECT COUNT(*) FROM mapping_codes m LEFT JOIN known_codes k ON k.code = m.rxcui
                WHERE k.code IS NULL),
            (SELECT COUNT(*) FROM mapping_aliases),
            (SELECT COUNT(*) FROM mapping_aliases m
                JOIN known_aliases k ON k.rxcui = m.rxcui AND k.normalized = m.normalized),
            (SELECT COUNT(*) FROM mapping_aliases m
                JOIN known_codes c ON c.code = m.rxcui
                LEFT JOIN known_aliases a
                  ON a.rxcui = m.rxcui AND a.normalized = m.normalized
                WHERE a.rxcui IS NULL)
        """
    ).fetchone()
    return {
        "mapping_rxcui_count": int(row[0]),
        "known_rxcui_count": int(row[1]),
        "unknown_rxcui_count": int(row[2]),
        "mapping_alias_pair_count": int(row[3]),
        "existing_alias_pair_count": int(row[4]),
        "review_alias_proposal_count": int(row[5]),
    }


def _missing_alias_proposals(
    connection: sqlite3.Connection,
    mapping_metadata: Mapping[str, str],
) -> Iterator[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            m.rxcui,
            m.normalized,
            m.rxstring,
            m.rxtty,
            COUNT(DISTINCT m.set_id || ':' || m.spl_version) AS support_count
        FROM mapping_rows m
        WHERE EXISTS (
            SELECT 1 FROM terminology.concepts c
            WHERE c.code_system = 'RxNorm' AND c.code = m.rxcui
        )
        AND NOT EXISTS (
            SELECT 1
            FROM terminology.aliases a
            JOIN terminology.concepts c ON c.concept_id = a.concept_id
            WHERE c.code_system = 'RxNorm'
              AND c.code = m.rxcui
              AND a.normalized = m.normalized
        )
        GROUP BY m.rxcui, m.normalized, m.rxstring, m.rxtty
        ORDER BY m.rxcui, m.normalized, m.rxstring, m.rxtty
        """
    )
    for key, grouped_rows in groupby(rows, key=lambda row: (str(row[0]), str(row[1]))):
        rxcui, normalized = key
        surfaces: dict[str, set[str]] = {}
        support_count = 0
        for _, _, surface, tty, support in grouped_rows:
            surfaces.setdefault(str(surface), set()).add(str(tty))
            support_count = max(support_count, int(support))
        identity = f"{rxcui}\0{normalized}".encode("utf-8")
        yield {
            "proposal_id": (
                f"dailymed-rxnorm-alias:{hashlib.sha256(identity).hexdigest()[:24]}"
            ),
            "code_system": "RxNorm",
            "code": rxcui,
            "normalized_alias": normalized,
            "surface_variants": [
                {"surface": surface, "ttys": sorted(ttys)}
                for surface, ttys in sorted(surfaces.items())
            ],
            "supporting_set_version_count": support_count,
            "source": "DailyMed SPL-RxNorm mapping",
            "source_version": mapping_metadata.get("source_version", ""),
            "source_sha256": mapping_metadata.get("source_sha256", ""),
            "review_status": "review_required",
            "recommended_use": "rxnorm_alias_overlay_after_review",
        }


def _absent_code_samples(
    connection: sqlite3.Connection,
    *,
    limit: int,
) -> list[dict[str, str]]:
    rows = connection.execute(
        """
        SELECT m.rxcui, MIN(m.rxstring)
        FROM mapping_rows m
        WHERE NOT EXISTS (
            SELECT 1 FROM terminology.concepts c
            WHERE c.code_system = 'RxNorm' AND c.code = m.rxcui
        )
        GROUP BY m.rxcui
        ORDER BY CAST(m.rxcui AS INTEGER), m.rxcui
        LIMIT ?
        """,
        (limit,),
    )
    return [{"rxcui": str(rxcui), "example_rxstring": str(text)} for rxcui, text in rows]


def _load_archive(
    connection: sqlite3.Connection,
    payload: bytes,
) -> tuple[int, Counter[str]]:
    input_count = 0
    tty_counts: Counter[str] = Counter()
    batch: list[tuple[str, str, str, str, str, str]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) != 1 or members[0].filename != _EXPECTED_MEMBER:
            raise ValueError("DailyMed mapping archive has an unexpected member layout")
        if members[0].file_size > _MAX_MEMBER_BYTES:
            raise ValueError("DailyMed mapping member exceeds the uncompressed size limit")
        with archive.open(members[0]) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"), delimiter="|")
            if tuple(reader.fieldnames or ()) != _EXPECTED_FIELDS:
                raise ValueError("DailyMed mapping header changed")
            for line_number, row in enumerate(reader, start=2):
                parsed = _parse_row(row, line_number=line_number)
                batch.append(parsed)
                input_count += 1
                tty_counts[parsed[5]] += 1
                if len(batch) >= 10_000:
                    _insert_batch(connection, batch)
                    batch.clear()
            _insert_batch(connection, batch)
    connection.commit()
    return input_count, tty_counts


def _parse_row(
    row: Mapping[str, str | None],
    *,
    line_number: int,
) -> tuple[str, str, str, str, str, str]:
    set_id = _required_field(row, "SETID", line_number).lower()
    spl_version = _required_field(row, "SPL_VERSION", line_number)
    rxcui = _required_field(row, "RXCUI", line_number)
    rxstring = _required_field(row, "RXSTRING", line_number)
    rxtty = _required_field(row, "RXTTY", line_number).upper()
    if not spl_version.isdigit() or not rxcui.isdigit():
        raise ValueError(f"DailyMed mapping line {line_number} has a non-numeric ID")
    if rxtty not in _SUPPORTED_TTYS:
        raise ValueError(f"DailyMed mapping line {line_number} has unknown TTY {rxtty!r}")
    normalized = normalize_for_match(rxstring)
    if not normalized:
        raise ValueError(f"DailyMed mapping line {line_number} has an empty normalized name")
    return set_id, spl_version, rxcui, rxstring, normalized, rxtty


def _required_field(
    row: Mapping[str, str | None],
    field_name: str,
    line_number: int,
) -> str:
    value = row.get(field_name)
    if value is None or not value.strip():
        raise ValueError(f"DailyMed mapping line {line_number} lacks {field_name}")
    return value.strip()


def _insert_batch(
    connection: sqlite3.Connection,
    batch: list[tuple[str, str, str, str, str, str]],
) -> None:
    if not batch:
        return
    connection.executemany(
        "INSERT OR IGNORE INTO mapping_rows VALUES (?, ?, ?, ?, ?, ?)",
        batch,
    )


def _finalize_index(
    connection: sqlite3.Connection,
    artifact: SourceArtifact,
    input_count: int,
) -> None:
    connection.executescript(
        """
        CREATE INDEX mapping_rows_by_rxcui ON mapping_rows(rxcui);
        CREATE INDEX mapping_rows_by_normalized ON mapping_rows(normalized, rxcui);
        ANALYZE;
        """
    )
    metadata = {
        "schema_version": _INDEX_SCHEMA_VERSION,
        "source_artifact_id": artifact.artifact_id,
        "source_sha256": artifact.object.sha256,
        "source_version": artifact.source_version,
        "input_row_count": str(input_count),
        "normalization_version": NORMALIZATION_CONTRACT_VERSION,
    }
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        sorted(metadata.items()),
    )
    connection.commit()


def _mapping_records(
    connection: sqlite3.Connection,
    artifact: SourceArtifact,
) -> Iterator[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT set_id, spl_version, rxcui, rxstring, normalized, rxtty
        FROM mapping_rows
        ORDER BY set_id, spl_version, rxcui, rxstring, rxtty
        """
    )
    for key, grouped_rows in groupby(rows, key=lambda row: (row[0], row[1], row[2])):
        set_id, spl_version, rxcui = (str(value) for value in key)
        aliases: dict[str, set[str]] = {}
        for _, _, _, rxstring, _, rxtty in grouped_rows:
            aliases.setdefault(str(rxstring), set()).add(str(rxtty))
        identity = f"{set_id}\0{spl_version}\0{rxcui}".encode("utf-8")
        yield {
            "mapping_id": f"dailymed-rxnorm:{hashlib.sha256(identity).hexdigest()[:24]}",
            "set_id": set_id,
            "spl_version": spl_version,
            "rxcui": rxcui,
            "rxstrings": [
                {"text": text, "ttys": sorted(ttys)}
                for text, ttys in sorted(aliases.items())
            ],
            "rxttys": sorted({tty for ttys in aliases.values() for tty in ttys}),
            "source_artifact_id": artifact.artifact_id,
            "source_version": artifact.source_version,
        }


def _concepts_from_rows(rows: Iterable[tuple[Any, ...]]) -> Iterator[DailyMedRxNormConcept]:
    for rxcui, grouped_rows in groupby(rows, key=lambda row: str(row[0])):
        strings: set[str] = set()
        ttys: set[str] = set()
        for _, rxstring, rxtty in grouped_rows:
            strings.add(str(rxstring))
            ttys.add(str(rxtty))
        yield DailyMedRxNormConcept(
            rxcui=rxcui,
            rxstrings=tuple(sorted(strings)),
            rxttys=tuple(sorted(ttys)),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
