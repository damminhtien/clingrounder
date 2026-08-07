"""Optional Parquet and DuckDB integration for frozen mining snapshots."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from clingrounder.mining.catalog import DuckDBMiningCatalog
from clingrounder.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    RelationProposal,
    ReviewStatus,
)
from clingrounder.mining.snapshot import SnapshotBuilder, SnapshotSplitConfig

pytestmark = pytest.mark.integration


def test_parquet_snapshot_registers_nonempty_and_empty_duckdb_views(
    tmp_path: Path,
) -> None:
    duckdb = importlib.import_module("duckdb")
    document = MinedDocument(
        document_id="doc-1",
        text="Bệnh nhân sốt.",
        language="vi",
        note_type="progress_note",
        source_artifact_id="fixture:artifact",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
    )
    snapshot_root = tmp_path / "snapshot"
    SnapshotBuilder(
        split_config=SnapshotSplitConfig(development_fraction=0.0)
    ).freeze(
        version="integration-v1",
        created_at="2026-07-18T00:00:00+00:00",
        output_dir=snapshot_root,
        documents=[document],
        write_parquet=True,
    )
    catalog_path = tmp_path / "catalog.duckdb"

    DuckDBMiningCatalog(catalog_path).register_snapshot(snapshot_root)

    connection = duckdb.connect(str(catalog_path), read_only=True)
    try:
        assert connection.execute(
            "SELECT document_id, split FROM snapshot_documents"
        ).fetchall() == [("doc-1", "train")]
        assert connection.execute("SELECT count(*) FROM snapshot_annotations").fetchone() == (
            0,
        )
        assert connection.execute("SELECT count(*) FROM snapshot_relations").fetchone() == (0,)
    finally:
        connection.close()


def test_parquet_snapshot_uses_stable_annotation_and_relation_schemas(
    tmp_path: Path,
) -> None:
    duckdb = importlib.import_module("duckdb")
    document = MinedDocument(
        document_id="doc-2",
        text="Sốt do cúm A.",
        language="vi",
        note_type="progress_note",
        source_artifact_id="fixture:artifact",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
    )
    symptom = AnnotationProposal(
        annotation_id="ann-symptom",
        document_id=document.document_id,
        span=(0, 3),
        text="Sốt",
        entity_type="SYMPTOM",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.GOLD,
        label_source="human",
        labeler_id="reviewer",
        review_status=ReviewStatus.ACCEPTED,
    )
    disease = AnnotationProposal(
        annotation_id="ann-disease",
        document_id=document.document_id,
        span=(7, 12),
        text="cúm A",
        entity_type="DISEASE",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.GOLD,
        label_source="human",
        labeler_id="reviewer",
        review_status=ReviewStatus.ACCEPTED,
    )
    relation = RelationProposal(
        relation_id="rel-1",
        document_id=document.document_id,
        head_annotation_id=disease.annotation_id,
        tail_annotation_id=symptom.annotation_id,
        relation_type="CAUSES",
        confidence=1.0,
        layer=AnnotationLayer.GOLD,
        label_source="human",
        labeler_id="reviewer",
        review_status=ReviewStatus.ACCEPTED,
        evidence_span=(0, len(document.text)),
    )
    snapshot_root = tmp_path / "snapshot_full"
    SnapshotBuilder(
        split_config=SnapshotSplitConfig(development_fraction=0.0)
    ).freeze(
        version="integration-full-v1",
        created_at="2026-07-18T00:00:00+00:00",
        output_dir=snapshot_root,
        documents=[document],
        annotations=[symptom, disease],
        relations=[relation],
        write_parquet=True,
    )
    catalog_path = tmp_path / "catalog.duckdb"

    DuckDBMiningCatalog(catalog_path).register_snapshot(snapshot_root)

    connection = duckdb.connect(str(catalog_path), read_only=True)
    try:
        assert connection.execute(
            "SELECT annotation_id, span_start, span_end "
            "FROM snapshot_full_annotations ORDER BY annotation_id"
        ).fetchall() == [("ann-disease", 7, 12), ("ann-symptom", 0, 3)]
        assert connection.execute(
            "SELECT relation_type, evidence_start, evidence_end "
            "FROM snapshot_full_relations"
        ).fetchall() == [("CAUSES", 0, len(document.text))]
    finally:
        connection.close()
