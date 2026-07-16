"""Optional Parquet snapshot writer and DuckDB catalog for large mined corpora."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from medical_kg_nlp.mining.records import AnnotationProposal, MinedDocument, RelationProposal

__all__ = ["DuckDBMiningCatalog", "ParquetSnapshotWriter"]


class ParquetSnapshotWriter:
    """Write deterministic, bounded-size Parquet shards using the optional data extra."""

    def __init__(self, root: str | Path, *, rows_per_shard: int = 50_000) -> None:
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self.root = Path(root)
        self.rows_per_shard = rows_per_shard

    def write(
        self,
        *,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
        relations: Sequence[RelationProposal] = (),
        splits: Mapping[str, str] | None = None,
    ) -> dict[str, list[str]]:
        try:
            pa = importlib.import_module("pyarrow")
            parquet = importlib.import_module("pyarrow.parquet")
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install medical-kg-nlp[data] to write Parquet snapshots") from error
        self.root.mkdir(parents=True, exist_ok=True)
        split_map = dict(splits or {})
        rows = {
            "documents": [
                {**document.to_dict(), "split": split_map.get(document.document_id, "train")}
                for document in sorted(documents, key=lambda item: item.document_id)
            ],
            "annotations": [
                proposal.to_dict()
                for proposal in sorted(annotations, key=lambda item: item.annotation_id)
            ],
            "relations": [
                _relation_dict(relation)
                for relation in sorted(relations, key=lambda item: item.relation_id)
            ],
        }
        output: dict[str, list[str]] = {}
        for table_name, table_rows in rows.items():
            output[table_name] = []
            table_root = self.root / table_name
            table_root.mkdir(parents=True, exist_ok=True)
            for shard_index, start in enumerate(range(0, len(table_rows), self.rows_per_shard)):
                shard_rows = table_rows[start : start + self.rows_per_shard]
                shard_path = table_root / f"part-{shard_index:05d}.parquet"
                table = pa.Table.from_pylist(shard_rows)
                parquet.write_table(table, shard_path, compression="zstd")
                output[table_name].append(str(shard_path))
        manifest_path = self.root / "tables.json"
        manifest_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


class DuckDBMiningCatalog:
    """Register immutable Parquet shards as read-only analytical views."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def register_snapshot(self, snapshot_root: str | Path) -> None:
        try:
            duckdb = importlib.import_module("duckdb")
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install medical-kg-nlp[data] to build a DuckDB catalog") from error
        root = Path(snapshot_root).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(self.path))
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS mining_snapshots "
                "(snapshot_root VARCHAR PRIMARY KEY, registered_at TIMESTAMP DEFAULT now())"
            )
            connection.execute(
                "INSERT OR REPLACE INTO mining_snapshots(snapshot_root) VALUES (?)",
                [str(root)],
            )
            for table_name in ("documents", "annotations", "relations"):
                pattern = str(root / table_name / "*.parquet")
                view_name = _safe_view_name(root.name, table_name)
                connection.execute(
                    f'CREATE OR REPLACE VIEW "{view_name}" AS SELECT * FROM read_parquet(?)',
                    [pattern],
                )
        finally:
            connection.close()


def _safe_view_name(snapshot_name: str, table_name: str) -> str:
    safe_snapshot = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in snapshot_name
    )
    return f"{safe_snapshot}_{table_name}"


def _relation_dict(relation: RelationProposal) -> dict[str, Any]:
    return {
        "relation_id": relation.relation_id,
        "document_id": relation.document_id,
        "head_annotation_id": relation.head_annotation_id,
        "tail_annotation_id": relation.tail_annotation_id,
        "relation_type": relation.relation_type,
        "confidence": relation.confidence,
        "layer": relation.layer.value,
        "label_source": relation.label_source,
        "evidence_span": list(relation.evidence_span) if relation.evidence_span else None,
        "metadata": dict(sorted(relation.metadata.items())),
    }
