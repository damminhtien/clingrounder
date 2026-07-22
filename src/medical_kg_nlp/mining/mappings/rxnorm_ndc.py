"""Compile and query the official RxNorm NDC attribute index.

DailyMed identifies a marketed product with the first two NDC segments, while
RxNorm stores package-level 11-digit NDC values in ``RXNSAT.RRF``.  This module
normalizes both representations into a nine-digit product prefix and preserves
the package rows as canonical JSONL.  The SQLite file is a rebuildable query
cache, never the source of truth.
"""

from __future__ import annotations

import io
import os
import sqlite3
import tempfile
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.utils.hashing import sha256_file

__all__ = [
    "RxNormNdcRepository",
    "compile_rxnorm_ndc_index",
    "normalize_ndc11",
    "normalize_ndc_product_prefix",
]

_INDEX_SCHEMA_VERSION = "rxnorm-ndc-index.v1"
_NORMALIZATION_VERSION = "fda-ndc11-product-prefix.v1"
_DEFAULT_MEMBER = "rrf/RXNSAT.RRF"
_MAX_MEMBER_BYTES = 1024 * 1024 * 1024


class RxNormNdcRepository:
    """Thread-local, read-only lookup over package and product NDC keys."""

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
            raise ValueError(f"Unsupported RxNorm NDC index: {self.path}")
        if metadata.get("normalization_version") != _NORMALIZATION_VERSION:
            raise ValueError(f"Unsupported RxNorm NDC normalization: {self.path}")
        if (
            expected_source_sha256 is not None
            and metadata.get("source_sha256") != expected_source_sha256
        ):
            raise ValueError("RxNorm NDC source fingerprint changed")

    @property
    def metadata(self) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in self._connection().execute(
                "SELECT key, value FROM metadata ORDER BY key"
            )
        }

    def lookup(self, ndc: str) -> tuple[str, ...]:
        """Return active RxCUIs for one package NDC or DailyMed product code."""

        compact = ndc.strip()
        if compact.isdigit() and len(compact) == 11:
            rows = self._connection().execute(
                "SELECT rxcui FROM ndc_rows WHERE ndc11 = ? ORDER BY rxcui",
                (compact,),
            )
        else:
            prefix = normalize_ndc_product_prefix(compact)
            rows = self._connection().execute(
                """
                SELECT DISTINCT rxcui
                FROM ndc_rows
                WHERE product_prefix = ?
                ORDER BY rxcui
                """,
                (prefix,),
            )
        return tuple(str(row[0]) for row in rows)

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


def normalize_ndc_product_prefix(value: str) -> str:
    """Normalize a two- or three-segment NDC into its nine-digit product key."""

    parts = tuple(part.strip() for part in value.strip().split("-"))
    if len(parts) == 3:
        return normalize_ndc11(value)[:9]
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Invalid NDC product code {value!r}")
    labeler, product = parts
    if (len(labeler), len(product)) == (4, 4):
        return f"0{labeler}{product}"
    if (len(labeler), len(product)) == (5, 3):
        return f"{labeler}0{product}"
    if (len(labeler), len(product)) == (5, 4):
        return f"{labeler}{product}"
    raise ValueError(f"Unsupported NDC product segment shape {value!r}")


def normalize_ndc11(value: str) -> str:
    """Normalize a package NDC using the FDA 4-4-2, 5-3-2, or 5-4-1 rule."""

    compact = value.strip()
    if compact.isdigit() and len(compact) == 11:
        return compact
    parts = tuple(part.strip() for part in compact.split("-"))
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Invalid package NDC {value!r}")
    labeler, product, package = parts
    shape = (len(labeler), len(product), len(package))
    if shape == (4, 4, 2):
        return f"0{labeler}{product}{package}"
    if shape == (5, 3, 2):
        return f"{labeler}0{product}{package}"
    if shape == (5, 4, 1):
        return f"{labeler}{product}0{package}"
    if shape == (5, 4, 2):
        return "".join(parts)
    raise ValueError(f"Unsupported package NDC segment shape {value!r}")


def compile_rxnorm_ndc_index(
    archive_path: str | Path,
    *,
    source_version: str,
    expected_source_sha256: str,
    output_path: str | Path,
    index_path: str | Path,
    report_path: str | Path,
    archive_member: str = _DEFAULT_MEMBER,
) -> dict[str, Any]:
    """Stream active RxNorm NDC rows into canonical JSONL and an atomic index."""

    archive = Path(archive_path)
    actual_source_sha256 = sha256_file(archive)
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            "RxNorm source fingerprint changed: "
            f"expected {expected_source_sha256}, observed {actual_source_sha256}"
        )
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
        counts = _load_rxnsat(connection, archive, archive_member=archive_member)
        unique_row_count = int(
            connection.execute("SELECT COUNT(*) FROM ndc_rows").fetchone()[0]
        )
        unique_product_count = int(
            connection.execute(
                "SELECT COUNT(DISTINCT product_prefix) FROM ndc_rows"
            ).fetchone()[0]
        )
        _write_metadata(
            connection,
            source_version=source_version,
            source_sha256=actual_source_sha256,
            archive_member=archive_member,
        )
        connection.execute(
            "CREATE INDEX ndc_rows_by_product ON ndc_rows(product_prefix, rxcui)"
        )
        connection.execute("CREATE INDEX ndc_rows_by_rxcui ON ndc_rows(rxcui)")
        connection.execute("ANALYZE")
        connection.commit()
        output_sha256 = write_jsonl(output, _iter_rows(connection, source_version))
        connection.close()
        connection = None
        # SCALING: readers observe the old complete index or the new complete index.
        os.replace(temporary_index, index)
        report: dict[str, Any] = {
            "schema_version": "rxnorm-ndc-compilation-report.v1",
            "path_base": "report_directory",
            "source_version": source_version,
            "source_sha256": actual_source_sha256,
            "archive_member": archive_member,
            **counts,
            "unique_row_count": unique_row_count,
            "duplicate_active_row_count": counts["active_ndc_row_count"]
            - unique_row_count,
            "unique_product_count": unique_product_count,
            "output": _relative_path(output, report_target.parent),
            "output_sha256": output_sha256,
            "index": _relative_path(index, report_target.parent),
            "index_sha256": sha256_file(index),
        }
        write_json(report_target, report)
        return report
    except BaseException:
        if connection is not None:
            connection.close()
        temporary_index.unlink(missing_ok=True)
        raise


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
    )
    connection.execute(
        """
        CREATE TABLE ndc_rows (
            ndc11 TEXT NOT NULL,
            product_prefix TEXT NOT NULL,
            rxcui TEXT NOT NULL,
            PRIMARY KEY (ndc11, rxcui)
        ) WITHOUT ROWID
        """
    )


def _load_rxnsat(
    connection: sqlite3.Connection,
    archive: Path,
    *,
    archive_member: str,
) -> dict[str, int]:
    total_row_count = 0
    active_ndc_row_count = 0
    invalid_ndc_row_count = 0
    batch: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(archive) as bundle:
        info = bundle.getinfo(archive_member)
        if info.file_size > _MAX_MEMBER_BYTES:
            raise ValueError("RxNorm RXNSAT member exceeds the decompressed size limit")
        with bundle.open(info) as raw, io.TextIOWrapper(raw, encoding="utf-8") as lines:
            for line in lines:
                total_row_count += 1
                fields = line.rstrip("\n").split("|")
                if len(fields) < 13:
                    continue
                rxcui = fields[0].strip()
                attribute = fields[8].strip().upper()
                source = fields[9].strip().upper()
                ndc = fields[10].strip()
                suppress = fields[11].strip().upper()
                if attribute != "NDC" or source != "RXNORM" or suppress in {"Y", "O"}:
                    continue
                active_ndc_row_count += 1
                if not rxcui.isdigit() or not ndc.isdigit() or len(ndc) != 11:
                    invalid_ndc_row_count += 1
                    continue
                batch.append((ndc, ndc[:9], rxcui))
                if len(batch) >= 10_000:
                    connection.executemany(
                        "INSERT OR IGNORE INTO ndc_rows VALUES (?, ?, ?)", batch
                    )
                    batch = []
        if batch:
            connection.executemany(
                "INSERT OR IGNORE INTO ndc_rows VALUES (?, ?, ?)", batch
            )
    connection.commit()
    return {
        "total_rxnsat_row_count": total_row_count,
        "active_ndc_row_count": active_ndc_row_count,
        "invalid_ndc_row_count": invalid_ndc_row_count,
    }


def _write_metadata(
    connection: sqlite3.Connection,
    *,
    source_version: str,
    source_sha256: str,
    archive_member: str,
) -> None:
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        (
            ("schema_version", _INDEX_SCHEMA_VERSION),
            ("normalization_version", _NORMALIZATION_VERSION),
            ("source_version", source_version),
            ("source_sha256", source_sha256),
            ("archive_member", archive_member),
        ),
    )


def _iter_rows(
    connection: sqlite3.Connection,
    source_version: str,
) -> Iterator[dict[str, str]]:
    rows = connection.execute(
        "SELECT product_prefix, ndc11, rxcui FROM ndc_rows ORDER BY product_prefix, ndc11, rxcui"
    )
    for product_prefix, ndc11, rxcui in rows:
        yield {
            "product_prefix": str(product_prefix),
            "ndc11": str(ndc11),
            "rxcui": str(rxcui),
            "source_version": source_version,
        }


def _relative_path(path: Path, base: Path) -> str:
    # INVARIANT: reports must remain valid after moving the repository.
    return Path(os.path.relpath(path.resolve(), start=base.resolve())).as_posix()
