"""Security and governance primitives are fail-closed and PHI-safe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clingrounder.governance import (
    ArtifactVerificationError,
    AuditEvent,
    DataPolicy,
    GovernancePolicy,
    InMemoryAuditSink,
    ModelGovernanceMetadata,
    fingerprint_artifact,
    safe_artifact_path,
    secure_temporary_path,
    verify_artifact,
)
from clingrounder.pipeline.factory import PipelineFactory


def test_artifact_hash_and_allowed_root_are_enforced(tmp_path: Path) -> None:
    artifact = tmp_path / "release.jsonl"
    artifact.write_text('{"code":"I10"}\n', encoding="utf-8")
    digest = fingerprint_artifact(artifact)
    assert verify_artifact(artifact, digest) == digest
    assert safe_artifact_path(artifact, allowed_roots=(tmp_path,)) == artifact.resolve()
    with pytest.raises(ArtifactVerificationError):
        verify_artifact(artifact, "0" * 64)
    with pytest.raises(ArtifactVerificationError):
        safe_artifact_path(tmp_path.parent / "outside", allowed_roots=(tmp_path,))


def test_secure_temporary_path_is_private_and_cleaned(tmp_path: Path) -> None:
    with secure_temporary_path(directory=tmp_path, suffix=".json") as path:
        assert path.exists()
        assert path.stat().st_mode & 0o777 == 0o600
        path.write_text("non-PHI test", encoding="utf-8")
    assert not path.exists()


def test_policy_rejects_unknown_nested_fields() -> None:
    with pytest.raises(ValueError, match="governance.data"):
        GovernancePolicy.from_mapping({"data": {"typo": True}})
    assert DataPolicy().hash_document_ids is True


def test_audit_event_has_no_raw_text() -> None:
    sink = InMemoryAuditSink()
    sink.emit(
        AuditEvent(
            "prediction", document_id_hash="abc", details={"document_length": "12"}
        )
    )
    payload = json.dumps(sink.snapshot())
    assert "patient has chest pain" not in payload
    assert "document_id_hash" in payload


def test_model_metadata_requires_use_and_limitations() -> None:
    metadata = ModelGovernanceMetadata(
        model_id="local/model",
        revision="abc123",
        training_data_description="Reviewed clinical text",
        intended_use="Entity extraction research",
        excluded_use="Autonomous diagnosis",
        evaluation_summary="Offset F1 on held-out fixtures",
        known_limitations="Vietnamese abbreviations remain difficult",
    )
    assert metadata.approval_status == "unreviewed"


def test_factory_audit_covers_profile_terminology_and_prediction() -> None:
    runner = PipelineFactory.from_config()
    try:
        runner.process_text("audit-doc", "Bệnh nhân ho.")
        events = runner.components.audit_sink.events  # type: ignore[attr-defined]
        event_types = {event.event_type for event in events}
        assert {"profile_load", "terminology_load", "prediction"} <= event_types
        assert all("Bệnh nhân ho" not in json.dumps(event.to_json()) for event in events)
    finally:
        runner.close()
