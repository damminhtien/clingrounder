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
                _document_row(
                    document,
                    split=split_map.get(document.document_id, "train"),
                )
                for document in sorted(documents, key=lambda item: item.document_id)
            ],
            "annotations": [
                _annotation_row(proposal)
                for proposal in sorted(annotations, key=lambda item: item.annotation_id)
            ],
            "relations": [
                _relation_row(relation)
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
                # SCALING: explicit scalar schemas avoid cross-shard drift from empty lists/maps.
                table = pa.Table.from_pylist(
                    shard_rows,
                    schema=_parquet_schema(pa, table_name),
                )
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
                view_name = _safe_view_name(root.name, table_name)
                shards = sorted((root / table_name).glob("*.parquet"))
                if shards:
                    shard_paths = ", ".join(
                        _duckdb_string_literal(str(path)) for path in shards
                    )
                    connection.execute(
                        f'CREATE OR REPLACE VIEW "{view_name}" AS '
                        f"SELECT * FROM read_parquet([{shard_paths}])"
                    )
                else:
                    # SCALING: empty tables remain queryable without manufacturing empty shards.
                    connection.execute(_empty_view_sql(view_name, table_name))
        finally:
            connection.close()


def _safe_view_name(snapshot_name: str, table_name: str) -> str:
    safe_snapshot = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in snapshot_name
    )
    return f"{safe_snapshot}_{table_name}"


def _empty_view_sql(view_name: str, table_name: str) -> str:
    columns = {
        "documents": (
            ("document_id", "VARCHAR"),
            ("text", "VARCHAR"),
            ("text_sha256", "VARCHAR"),
            ("language", "VARCHAR"),
            ("note_type", "VARCHAR"),
            ("source_artifact_id", "VARCHAR"),
            ("access_class", "VARCHAR"),
            ("redistribution", "VARCHAR"),
            ("hosted_processing_allowed", "BOOLEAN"),
            ("parent_document_id", "VARCHAR"),
            ("group_ids_json", "VARCHAR"),
            ("metadata_json", "VARCHAR"),
            ("split", "VARCHAR"),
        ),
        "annotations": (
            ("annotation_id", "VARCHAR"),
            ("document_id", "VARCHAR"),
            ("span_start", "BIGINT"),
            ("span_end", "BIGINT"),
            ("text", "VARCHAR"),
            ("entity_type", "VARCHAR"),
            ("assertions_json", "VARCHAR"),
            ("concepts_json", "VARCHAR"),
            ("confidence", "DOUBLE"),
            ("layer", "VARCHAR"),
            ("label_source", "VARCHAR"),
            ("labeler_id", "VARCHAR"),
            ("review_status", "VARCHAR"),
            ("source_label", "VARCHAR"),
            ("model_revision", "VARCHAR"),
            ("prompt_hash", "VARCHAR"),
            ("metadata_json", "VARCHAR"),
        ),
        "relations": (
            ("relation_id", "VARCHAR"),
            ("document_id", "VARCHAR"),
            ("head_annotation_id", "VARCHAR"),
            ("tail_annotation_id", "VARCHAR"),
            ("relation_type", "VARCHAR"),
            ("confidence", "DOUBLE"),
            ("layer", "VARCHAR"),
            ("label_source", "VARCHAR"),
            ("evidence_start", "BIGINT"),
            ("evidence_end", "BIGINT"),
            ("metadata_json", "VARCHAR"),
        ),
    }[table_name]
    projection = ", ".join(
        f'CAST(NULL AS {data_type}) AS "{name}"' for name, data_type in columns
    )
    return f'CREATE OR REPLACE VIEW "{view_name}" AS SELECT {projection} WHERE FALSE'


def _document_row(document: MinedDocument, *, split: str) -> dict[str, Any]:
    raw = document.to_dict()
    return {
        "document_id": raw["document_id"],
        "text": raw["text"],
        "text_sha256": raw["text_sha256"],
        "language": raw["language"],
        "note_type": raw["note_type"],
        "source_artifact_id": raw["source_artifact_id"],
        "access_class": raw["access_class"],
        "redistribution": raw["redistribution"],
        "hosted_processing_allowed": raw["hosted_processing_allowed"],
        "parent_document_id": raw["parent_document_id"],
        "group_ids_json": _canonical_json(raw["group_ids"]),
        "metadata_json": _canonical_json(raw["metadata"]),
        "split": split,
    }


def _annotation_row(proposal: AnnotationProposal) -> dict[str, Any]:
    raw = proposal.to_dict()
    span = raw["span"]
    return {
        "annotation_id": raw["annotation_id"],
        "document_id": raw["document_id"],
        "span_start": span[0],
        "span_end": span[1],
        "text": raw["text"],
        "entity_type": raw["entity_type"],
        "assertions_json": _canonical_json(raw["assertions"]),
        "concepts_json": _canonical_json(raw["concepts"]),
        "confidence": raw["confidence"],
        "layer": raw["layer"],
        "label_source": raw["label_source"],
        "labeler_id": raw["labeler_id"],
        "review_status": raw["review_status"],
        "source_label": raw["source_label"],
        "model_revision": raw["model_revision"],
        "prompt_hash": raw["prompt_hash"],
        "metadata_json": _canonical_json(raw["metadata"]),
    }


def _relation_row(relation: RelationProposal) -> dict[str, Any]:
    raw = relation.to_dict()
    evidence = raw["evidence_span"]
    return {
        "relation_id": raw["relation_id"],
        "document_id": raw["document_id"],
        "head_annotation_id": raw["head_annotation_id"],
        "tail_annotation_id": raw["tail_annotation_id"],
        "relation_type": raw["relation_type"],
        "confidence": raw["confidence"],
        "layer": raw["layer"],
        "label_source": raw["label_source"],
        "evidence_start": None if evidence is None else evidence[0],
        "evidence_end": None if evidence is None else evidence[1],
        "metadata_json": _canonical_json(raw["metadata"]),
    }


def _parquet_schema(pa: Any, table_name: str) -> Any:
    string = pa.string()
    integer = pa.int64()
    schemas = {
        "documents": [
            ("document_id", string),
            ("text", string),
            ("text_sha256", string),
            ("language", string),
            ("note_type", string),
            ("source_artifact_id", string),
            ("access_class", string),
            ("redistribution", string),
            ("hosted_processing_allowed", pa.bool_()),
            ("parent_document_id", string),
            ("group_ids_json", string),
            ("metadata_json", string),
            ("split", string),
        ],
        "annotations": [
            ("annotation_id", string),
            ("document_id", string),
            ("span_start", integer),
            ("span_end", integer),
            ("text", string),
            ("entity_type", string),
            ("assertions_json", string),
            ("concepts_json", string),
            ("confidence", pa.float64()),
            ("layer", string),
            ("label_source", string),
            ("labeler_id", string),
            ("review_status", string),
            ("source_label", string),
            ("model_revision", string),
            ("prompt_hash", string),
            ("metadata_json", string),
        ],
        "relations": [
            ("relation_id", string),
            ("document_id", string),
            ("head_annotation_id", string),
            ("tail_annotation_id", string),
            ("relation_type", string),
            ("confidence", pa.float64()),
            ("layer", string),
            ("label_source", string),
            ("evidence_start", integer),
            ("evidence_end", integer),
            ("metadata_json", string),
        ],
    }
    return pa.schema(schemas[table_name])


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _duckdb_string_literal(value: str) -> str:
    # DuckDB does not permit prepared parameters in CREATE VIEW; quote generated local paths.
    return "'" + value.replace("'", "''") + "'"
