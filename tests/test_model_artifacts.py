from dataclasses import replace
from pathlib import Path

import pytest

from clingrounder.governance import (
    ModelArtifactManifest,
    ModelGovernanceMetadata,
    load_model_artifact_manifest,
    verify_model_artifact,
)
from clingrounder.governance.artifacts import fingerprint_artifact


def _manifest(model_sha: str, config_sha: str) -> ModelArtifactManifest:
    governance = ModelGovernanceMetadata(
        model_id="local/vi-ner",
        revision="a" * 40,
        training_data_description="Synthetic public fixture for contract tests.",
        intended_use="Research entity extraction.",
        excluded_use="Clinical decision support.",
        evaluation_summary="Public pilot only.",
        known_limitations="Small synthetic fixture.",
    )
    return ModelArtifactManifest(
        artifact_id="vi-ner-test",
        task="entity-extraction",
        model_id="local/vi-ner",
        revision="a" * 40,
        artifact_sha256=model_sha,
        governance=governance,
        training_data_fingerprints=("b" * 64,),
        config_sha256=config_sha,
        metrics=(("entity_f1", 0.8),),
    )


def test_model_manifest_round_trips_and_verifies_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"hidden": 8}\n', encoding="utf-8")
    (model_dir / "weights.safetensors").write_bytes(b"weights")
    manifest = _manifest(fingerprint_artifact(model_dir), "c" * 64)
    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)

    loaded = load_model_artifact_manifest(manifest_path)
    assert loaded == manifest
    assert verify_model_artifact(model_dir, loaded) == manifest.artifact_sha256


def test_model_manifest_rejects_tampering(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight = model_dir / "weights.bin"
    weight.write_bytes(b"original")
    manifest = _manifest(fingerprint_artifact(model_dir), "c" * 64)
    weight.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_model_artifact(model_dir, manifest)


def test_approved_gate_is_explicit(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"weights")
    manifest = _manifest(fingerprint_artifact(model_dir), "c" * 64)

    with pytest.raises(ValueError, match="not approved"):
        verify_model_artifact(model_dir, manifest, require_approved=True)


def test_approved_gate_requires_tokenizer_and_runtime_pins(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"weights")
    manifest = _manifest(fingerprint_artifact(model_dir), "c" * 64)
    approved = replace(
        manifest,
        governance=replace(manifest.governance, approval_status="approved"),
    )

    with pytest.raises(ValueError, match="tokenizer and runtime_profile"):
        verify_model_artifact(model_dir, approved, require_approved=True)


def test_approved_manifest_round_trips_pinned_runtime_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"weights")
    manifest = _manifest(fingerprint_artifact(model_dir), "c" * 64)
    approved = replace(
        manifest,
        governance=replace(manifest.governance, approval_status="approved"),
        tokenizer_id="FacebookAI/xlm-roberta-base",
        tokenizer_revision="b" * 40,
        runtime_profile="vi-clinical-small@2026.08",
    )
    path = tmp_path / "manifest.json"
    approved.write(path)

    loaded = load_model_artifact_manifest(path)

    assert loaded == approved
    assert verify_model_artifact(model_dir, loaded, require_approved=True) == approved.artifact_sha256


def test_model_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema_version":"clingrounder.model-artifact.v1","unexpected":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown model artifact fields"):
        load_model_artifact_manifest(path)
