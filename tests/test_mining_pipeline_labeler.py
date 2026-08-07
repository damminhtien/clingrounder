"""Contract tests for local pipeline proposals used by data mining."""

from __future__ import annotations

from dataclasses import dataclass

from clingrounder.mining.labelers.pipeline import LocalPipelineProposalLabeler
from clingrounder.mining.records import AccessClass, MinedDocument, RedistributionPolicy
from clingrounder.schema.annotation import (
    AssertionFeatures,
    CandidateConcept,
    EntityAnnotation,
)
from clingrounder.schema.output import ClinicalPrediction
from clingrounder.schema.types import AssertionStatus, CodeSystem, EntityType


@dataclass
class _FakeRunner:
    """Small deterministic runner stand-in; no model dependency is needed for this test."""

    def process_text(self, document_id: str, text: str, metadata: dict[str, str] | None = None):
        del metadata
        start = text.index("tăng huyết áp")
        entity = EntityAnnotation(
            id="E1",
            span=(start, start + len("tăng huyết áp")),
            text="tăng huyết áp",
            normalized_text="tăng huyết áp",
            type=EntityType.DISEASE,
            assertion=AssertionStatus.HISTORICAL,
            assertion_features=AssertionFeatures(historical=True),
            code_system=CodeSystem.ICD10,
            code="I10",
            confidence=0.91,
            candidates=[
                CandidateConcept(
                    code_system=CodeSystem.ICD10,
                    code="I10",
                    name="Essential hypertension",
                    retrieval_score=1.0,
                    emit_probability=0.0,
                    concept_id="ICD10:I10",
                    source="exact",
                    evidence_sources=("exact",),
                    matched_alias="tăng huyết áp",
                    qualified=True,
                    qualification_reason="exact",
                )
            ],
        )
        return ClinicalPrediction.from_text(document_id, text, [entity], [], "fake-pipeline")


def _document() -> MinedDocument:
    return MinedDocument(
        document_id="rare-1",
        text="Tiền sử: tăng huyết áp.",
        language="vi",
        note_type="case_report",
        source_artifact_id="artifact:rare",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
    )


def test_local_pipeline_labeler_preserves_offsets_and_provenance() -> None:
    labeler = LocalPipelineProposalLabeler(
        _FakeRunner(),
        labeler_id="fake-pipeline:v1",
        terminology_versions={"ICD-10": "TT06/2026"},
        model_revision="rules@v1",
    )

    proposals = tuple(labeler.propose((_document(),)))

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.text == "tăng huyết áp"
    assert proposal.span == (9, 22)
    assert proposal.assertions == ("HISTORICAL",)
    assert proposal.concepts[0].code == "I10"
    assert proposal.concepts[0].terminology_version == "TT06/2026"
    assert proposal.layer.value == "bronze"
    assert proposal.review_status.value == "proposed"
    assert proposal.model_revision == "rules@v1"
    proposal.validate_offsets(_document())
