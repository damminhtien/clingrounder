"""Snapshot determinism, leakage, challenge, and synthetic-ratio tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)
from medical_kg_nlp.mining.snapshot import SnapshotBuilder, SnapshotSplitConfig


def _document(
    document_id: str,
    text: str,
    *,
    source: str = "train_source",
    origin: str = "real",
    group_ids: tuple[str, ...] = (),
) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text=text,
        language="vi",
        note_type="progress_note",
        source_artifact_id=f"{source}:artifact",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
        group_ids=group_ids,
        metadata={"source_id": source, "origin": origin},
    )


def _gold(document: MinedDocument, annotation_id: str) -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document.document_id,
        span=(0, len(document.text)),
        text=document.text,
        entity_type="DISEASE",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.GOLD,
        label_source="human_review",
        labeler_id="reviewer-1",
        review_status=ReviewStatus.ACCEPTED,
    )


def test_snapshot_promotes_connected_duplicate_cluster_to_challenge(tmp_path: Path) -> None:
    held_out = _document("held", "Cùng một ca bệnh.", source="held_source")
    duplicate = _document("duplicate", "  CÙNG một ca bệnh.  ")
    train = _document("train", "Một hồ sơ khác.")
    documents = [held_out, duplicate, train]
    annotations = [_gold(document, f"ann-{document.document_id}") for document in documents]
    builder = SnapshotBuilder(
        split_config=SnapshotSplitConfig(
            development_fraction=0.0,
            challenge_sources=frozenset({"held_source"}),
        )
    )

    snapshot = builder.freeze(
        version="phase2-open-v1",
        created_at="2026-07-18T00:00:00+00:00",
        output_dir=tmp_path / "snapshot-a",
        documents=documents,
        annotations=annotations,
        source_fingerprints=("a" * 64,),
        write_parquet=False,
    )
    manifest = json.loads((tmp_path / "snapshot-a" / "manifest.json").read_text())

    assert manifest["splits"]["held"] == "challenge"
    assert manifest["splits"]["duplicate"] == "challenge"
    assert manifest["splits"]["train"] == "train"
    assert snapshot.split_counts == (("challenge", 2), ("train", 1))


def test_snapshot_manifest_is_deterministic_and_freeze_is_idempotent(tmp_path: Path) -> None:
    document = _document("doc", "Tăng huyết áp")
    annotation = _gold(document, "ann")
    builder = SnapshotBuilder(
        split_config=SnapshotSplitConfig(development_fraction=0.0)
    )
    arguments = {
        "version": "v1",
        "created_at": "2026-07-18T00:00:00+00:00",
        "documents": [document],
        "annotations": [annotation],
        "write_parquet": False,
    }

    first = builder.freeze(output_dir=tmp_path / "one", **arguments)
    repeated = builder.freeze(output_dir=tmp_path / "one", **arguments)
    second = builder.freeze(output_dir=tmp_path / "two", **arguments)

    assert first == repeated == second
    assert (tmp_path / "one" / "manifest.json").read_bytes() == (
        tmp_path / "two" / "manifest.json"
    ).read_bytes()
    with pytest.raises(FileExistsError, match="Immutable snapshot"):
        builder.freeze(
            output_dir=tmp_path / "one",
            **{**arguments, "version": "v2"},
        )


def test_snapshot_rejects_unreviewed_challenge_annotation(tmp_path: Path) -> None:
    document = _document("held", "Bệnh hiếm", source="held_source")
    annotation = _gold(document, "ann")
    annotation = replace(
        annotation,
        layer=AnnotationLayer.SILVER,
        review_status=ReviewStatus.PROPOSED,
    )
    builder = SnapshotBuilder(
        split_config=SnapshotSplitConfig(
            development_fraction=0.0,
            challenge_sources=frozenset({"held_source"}),
        )
    )

    with pytest.raises(ValueError, match="non_gold_challenge"):
        builder.freeze(
            version="v1",
            created_at="2026-07-18T00:00:00+00:00",
            output_dir=tmp_path / "snapshot",
            documents=[document],
            annotations=[annotation],
            write_parquet=False,
        )


def test_snapshot_rejects_excess_synthetic_training_documents(tmp_path: Path) -> None:
    real = _document("real", "Real document")
    synthetic = _document("synthetic", "Synthetic document", origin="synthetic")
    builder = SnapshotBuilder(
        split_config=SnapshotSplitConfig(
            development_fraction=0.0,
            max_synthetic_train_fraction=0.4,
        )
    )

    with pytest.raises(ValueError, match="synthetic_train_fraction"):
        builder.freeze(
            version="v1",
            created_at="2026-07-18T00:00:00+00:00",
            output_dir=tmp_path / "snapshot",
            documents=[real, synthetic],
            write_parquet=False,
        )


def test_snapshot_never_places_synthetic_document_in_challenge(tmp_path: Path) -> None:
    document = _document(
        "synthetic",
        "Synthetic challenge",
        source="held_source",
        origin="synthetic",
    )
    builder = SnapshotBuilder(
        split_config=SnapshotSplitConfig(
            development_fraction=0.0,
            challenge_sources=frozenset({"held_source"}),
        )
    )

    with pytest.raises(ValueError, match="synthetic_challenge_document"):
        builder.freeze(
            version="v1",
            created_at="2026-07-18T00:00:00+00:00",
            output_dir=tmp_path / "snapshot",
            documents=[document],
            write_parquet=False,
        )
