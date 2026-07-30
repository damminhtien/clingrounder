"""Registry, privacy, licensing, and mined-data quality tests."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from medical_kg_nlp.mining import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
    SourceArtifact,
    StoredObject,
)
from medical_kg_nlp.mining.policy import MiningQualityGate, SourcePolicyGate
from medical_kg_nlp.mining.registry import SourceRegistry, load_source_registry


def test_checked_in_mining_registry_is_strict_and_unique() -> None:
    registry = load_source_registry("data/sources/mining_registry.yaml")

    assert registry.schema_version == "medical-source-registry.v2"
    assert registry.by_id("vietbioner").license_id == "CC-BY-4.0"
    assert registry.by_id("vietbioner").parser_options["language"] == "vi"
    assert registry.by_id("mimic_iv_note").hosted_processing_allowed is False
    assert registry.by_id("pmc_oa").license_mode.value == "per_artifact"
    round2 = registry.by_id("phase1_round2_input")
    assert round2.access_class is AccessClass.AUTHORIZED_PRIVATE
    assert round2.redistribution is RedistributionPolicy.PROHIBITED
    assert round2.hosted_processing_allowed is True
    assert round2.allowed_uses == (
        "local_competition_inference",
        "hosted_competition_inference",
        "local_distribution_audit",
    )
    leaked = registry.by_id("phase1_part2_leaked_bundle")
    assert leaked.access_class is AccessClass.AUTHORIZED_PRIVATE
    assert leaked.redistribution is RedistributionPolicy.PROHIBITED
    assert leaked.hosted_processing_allowed is True
    assert leaked.parser_options["offset_coordinate_view"] == "crlf_to_lf_child_document"
    assert leaked.allowed_uses == (
        "local_supervised_training",
        "hosted_supervised_training",
        "distillation",
        "evaluation_diagnostics",
    )


def test_registry_rejects_hosted_processing_for_dua_source() -> None:
    payload = {
        "schema_version": "medical-source-registry.v2",
        "resources": [
            {
                "id": "private",
                "name": "Private",
                "category": "clinical_notes",
                "version": "1",
                "version_policy": "pinned",
                "access_class": "dua",
                "license_id": "dua",
                "license_url": "https://example.test/dua",
                "redistribution": "prohibited",
                "hosted_processing_allowed": True,
                "retention": "local_only",
                "connector": "local_archive",
                "parser": "notes",
                "allowed_uses": ["local_training"],
            }
        ],
    }

    with pytest.raises(ValidationError, match="restricted sources"):
        SourceRegistry.model_validate(payload)


def test_source_policy_gate_rejects_version_drift() -> None:
    registry = load_source_registry("data/sources/mining_registry.yaml")
    gate = SourcePolicyGate(registry)
    digest = hashlib.sha256(b"fixture").hexdigest()
    artifact = SourceArtifact(
        artifact_id="mimic-note",
        source_id="mimic_iv_note",
        source_version="2.1",
        source_uri="file:///private/mimic.csv.gz",
        object=StoredObject(digest, "file:///data/object", 7),
        media_type="text/csv",
        license_id="physionet-credentialed-health-data-1.5.0",
        access_class=AccessClass.DUA,
        redistribution=RedistributionPolicy.PROHIBITED,
        hosted_processing_allowed=False,
        retrieved_at="2026-07-16T00:00:00Z",
    )

    decision = gate.validate_artifact(artifact)

    assert decision.allowed is False
    assert "source_version_mismatch" in decision.reasons


def test_source_policy_gate_accepts_per_artifact_redistribution_policy() -> None:
    registry = load_source_registry("data/sources/mining_registry.yaml")
    gate = SourcePolicyGate(registry)
    source = registry.by_id("pmc_oa")
    artifact = SourceArtifact(
        artifact_id="pmc-article",
        source_id=source.id,
        source_version="oa-query-v1",
        source_uri="https://example.test/PMC42.json",
        object=StoredObject("a" * 64, "file:///data/object", 7),
        media_type="application/json",
        license_id="CC BY-NC-ND",
        access_class=source.access_class,
        redistribution=RedistributionPolicy.PROHIBITED,
        hosted_processing_allowed=source.hosted_processing_allowed,
        retrieved_at="2026-07-18T00:00:00Z",
    )

    decision = gate.validate_artifact(artifact)

    assert decision.allowed is True
    assert decision.reasons == ()


def test_source_policy_gate_requires_local_encrypted_dua_storage() -> None:
    registry = load_source_registry("data/sources/mining_registry.yaml")
    gate = SourcePolicyGate(registry)
    source = registry.by_id("mimic_iv_note")

    unencrypted = gate.artifact_storage(
        source,
        store_uri="/Volumes/private-medical-data",
        encrypted_at_rest=False,
    )
    remote = gate.artifact_storage(
        source,
        store_uri="s3://example/mimic",
        encrypted_at_rest=True,
    )
    allowed = gate.artifact_storage(
        source,
        store_uri="/Volumes/encrypted-medical-data",
        encrypted_at_rest=True,
    )

    assert "restricted_source_requires_encryption_at_rest" in unencrypted.reasons
    assert "local_only_source_requires_local_store" in remote.reasons
    assert allowed.allowed is True


def test_quality_gate_reports_offsets_unknown_concepts_and_synthetic_challenge() -> None:
    document = MinedDocument(
        document_id="doc-1",
        text="Tăng huyết áp",
        language="vi",
        note_type="problem_list",
        source_artifact_id="source-1",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
    )
    proposal = AnnotationProposal(
        annotation_id="ann-1",
        document_id="doc-1",
        span=(0, 4),
        text="Sai!",
        entity_type="DISEASE",
        assertions=(),
        concepts=(ConceptLink("ICD-10", "I10", "TT06-2026"),),
        confidence=1.0,
        layer=AnnotationLayer.CHALLENGE,
        label_source="human",
        labeler_id="reviewer",
        review_status=ReviewStatus.ACCEPTED,
        metadata={"origin": "synthetic"},
    )
    gate = MiningQualityGate(known_concepts=set())

    issues = gate.validate([document], [proposal])

    assert any(issue.startswith("offset:ann-1") for issue in issues)
    assert "synthetic_challenge:ann-1" in issues
    assert "unknown_concept:ann-1:ICD-10:I10" in issues
