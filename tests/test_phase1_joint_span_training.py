"""Tests for non-ML joint span verifier training input and artifact contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.joint_span import Phase1JointSpanLabel
from medical_kg_nlp.benchmarks.phase1.joint_span_training import (
    Phase1JointSpanTrainingConfig,
    inspect_phase1_joint_span_training_inputs,
    phase1_joint_span_training_family_fingerprint,
)
from medical_kg_nlp.utils.hashing import sha256_file


def test_inspect_joint_span_training_inputs_accepts_all_label_contracts(tmp_path: Path) -> None:
    dataset = tmp_path / "examples.jsonl"
    rows = [_row(index, label) for index, label in enumerate(Phase1JointSpanLabel)]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "phase1-joint-span-dataset.v1",
                "examples_sha256": sha256_file(dataset),
            }
        ),
        encoding="utf-8",
    )

    summary = inspect_phase1_joint_span_training_inputs(_config(dataset, manifest, tmp_path))

    assert summary.example_count == len(Phase1JointSpanLabel)
    assert summary.document_count == len(Phase1JointSpanLabel)
    assert summary.label_counts == {label.value: 1 for label in Phase1JointSpanLabel}


def test_joint_span_training_family_fingerprint_ignores_output_directory(tmp_path: Path) -> None:
    dataset = tmp_path / "examples.jsonl"
    rows = [_row(index, label) for index, label in enumerate(Phase1JointSpanLabel)]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "phase1-joint-span-dataset.v1",
                "examples_sha256": sha256_file(dataset),
            }
        ),
        encoding="utf-8",
    )

    first = _config(dataset, manifest, tmp_path)
    second = Phase1JointSpanTrainingConfig(
        dataset_path=dataset,
        dataset_manifest_path=manifest,
        output_dir=tmp_path / "other-output",
        model_id=first.model_id,
        revision=first.revision,
    )

    assert phase1_joint_span_training_family_fingerprint(
        first, inspect_phase1_joint_span_training_inputs(first)
    ) == phase1_joint_span_training_family_fingerprint(
        second, inspect_phase1_joint_span_training_inputs(second)
    )


def test_joint_span_training_rejects_manifest_hash_drift(tmp_path: Path) -> None:
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text(
        json.dumps(_row(0, Phase1JointSpanLabel.EXACT_DISEASE)) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "phase1-joint-span-dataset.v1",
                "examples_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        inspect_phase1_joint_span_training_inputs(_config(dataset, manifest, tmp_path))


def test_joint_span_training_requires_paired_initializer_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be paired"):
        Phase1JointSpanTrainingConfig(
            dataset_path=tmp_path / "examples.jsonl",
            dataset_manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "output",
            model_id="pinned/model",
            revision="pinned-revision",
            initialization_model_path=tmp_path / "initializer",
        )


def _config(dataset: Path, manifest: Path, root: Path) -> Phase1JointSpanTrainingConfig:
    return Phase1JointSpanTrainingConfig(
        dataset_path=dataset,
        dataset_manifest_path=manifest,
        output_dir=root / "output",
        model_id="pinned/model",
        revision="pinned-revision",
    )


def _row(index: int, label: Phase1JointSpanLabel) -> dict[str, object]:
    return {
        "variant_id": f"variant-{index}",
        "document_id": f"doc-{index}",
        "text": "entity",
        "position": [0, 6],
        "label": label.value,
        "source_dataset": "manual_gold",
        "cross_encoder_text": "[ENTITY] entity",
    }
