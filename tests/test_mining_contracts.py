"""Contracts and invariants for the task-neutral mining subsystem."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from medical_kg_nlp.mining import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    LocalArtifactStore,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)
from medical_kg_nlp.mining.io import load_source_artifacts


def test_local_artifact_store_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "data")

    first = store.put_stream(io.BytesIO(b"clinical text"), metadata={"source": "fixture"})
    second = store.put_stream(io.BytesIO(b"clinical text"), metadata={"source": "fixture"})

    assert first == second
    assert first.sha256 == hashlib.sha256(b"clinical text").hexdigest()
    assert first.uri == f"medical-kg-cas://sha256/{first.sha256}"
    assert str(tmp_path) not in first.uri
    assert store.exists(first.sha256)
    with store.open(first.sha256) as handle:
        assert handle.read() == b"clinical text"
    metadata = next((tmp_path / "data" / "metadata").rglob("*.json"))
    assert json.loads(metadata.read_text(encoding="utf-8"))["metadata"]["source"] == "fixture"


def test_legacy_source_manifest_uri_is_migrated_to_portable_cas(tmp_path: Path) -> None:
    digest = "a" * 64
    manifest = tmp_path / "artifacts.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "artifact_id": "fixture:artifact",
                "source_id": "fixture",
                "source_version": "v1",
                "source_uri": "https://example.test/source.zip",
                "object": {
                    "sha256": digest,
                    "uri": f"file://{tmp_path}/objects/{digest}",
                    "byte_size": 1,
                },
                "media_type": "application/zip",
                "license_id": "fixture-license",
                "access_class": "open",
                "redistribution": "allowed",
                "hosted_processing_allowed": True,
                "retrieved_at": "2026-07-22T00:00:00+00:00",
                "metadata": {},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = load_source_artifacts(manifest)[0]

    assert artifact.object.uri == f"medical-kg-cas://sha256/{digest}"


def test_annotation_proposal_validates_raw_offsets() -> None:
    document = MinedDocument(
        document_id="doc-1",
        text="Bệnh nhân không sốt.",
        language="vi",
        note_type="progress_note",
        source_artifact_id="artifact-1",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
    )
    proposal = AnnotationProposal(
        annotation_id="ann-1",
        document_id="doc-1",
        span=(16, 19),
        text="sốt",
        entity_type="SYMPTOM",
        assertions=("NEGATED",),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.GOLD,
        label_source="human",
        labeler_id="reviewer-1",
        review_status=ReviewStatus.ACCEPTED,
    )

    proposal.validate_offsets(document)

    invalid = AnnotationProposal(
        **{**proposal.__dict__, "annotation_id": "ann-2", "span": (15, 19)}
    )
    with pytest.raises(ValueError, match="Offset mismatch"):
        invalid.validate_offsets(document)


def test_restricted_documents_cannot_be_sent_to_hosted_labelers() -> None:
    with pytest.raises(ValueError, match="Restricted documents"):
        MinedDocument(
            document_id="doc-private",
            text="Deidentified note",
            language="en",
            note_type="discharge_summary",
            source_artifact_id="mimic-artifact",
            access_class=AccessClass.DUA,
            redistribution=RedistributionPolicy.PROHIBITED,
            hosted_processing_allowed=True,
        )


def test_concept_links_are_versioned_and_confidence_bounded() -> None:
    link = ConceptLink(
        code_system="ICD-10",
        code="I10",
        terminology_version="TT06-2026",
        confidence=0.95,
    )

    assert link.to_dict()["terminology_version"] == "TT06-2026"
    with pytest.raises(ValueError, match="between 0 and 1"):
        ConceptLink(
            code_system="ICD-10",
            code="I10",
            terminology_version="TT06-2026",
            confidence=1.1,
        )
