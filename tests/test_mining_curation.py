"""Tests for deduplication, coverage ranking, consensus, and review exchange."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

import pytest

from medical_kg_nlp.mining.coverage import CoverageCubePlanner, CoverageTarget
from medical_kg_nlp.mining.dedup import StableTextDeduplicator
from medical_kg_nlp.mining.labeling import (
    ConsensusProposalLabeler,
    PolicyAwareProposalLabelerAdapter,
)
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)
from medical_kg_nlp.mining.review import JsonlReviewBackend


def _document(document_id: str, text: str, **metadata: str) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text=text,
        language="vi",
        note_type="progress_note",
        source_artifact_id="fixture:artifact",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
        metadata=metadata,
    )


def _proposal(
    document: MinedDocument,
    *,
    annotation_id: str,
    labeler_id: str,
    entity_type: str = "SYMPTOM",
    review_status: ReviewStatus = ReviewStatus.PROPOSED,
) -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document.document_id,
        span=(0, len(document.text)),
        text=document.text,
        entity_type=entity_type,
        assertions=(),
        concepts=(),
        confidence=0.9,
        layer=AnnotationLayer.BRONZE,
        label_source="fixture",
        labeler_id=labeler_id,
        review_status=review_status,
    )


class FixtureLabeler:
    def __init__(self, proposals: Sequence[AnnotationProposal]) -> None:
        self.proposals = proposals

    def propose(self, documents: Sequence[MinedDocument]) -> Iterable[AnnotationProposal]:
        allowed = {document.document_id for document in documents}
        return (proposal for proposal in self.proposals if proposal.document_id in allowed)


def test_deduplicator_groups_normalized_and_near_duplicate_text() -> None:
    documents = [
        _document("a", "Bệnh nhân đau ngực nặng hôm nay."),
        _document("b", "  BỆNH NHÂN đau ngực nặng hôm nay. "),
        _document("c", "Bệnh nhân đau ngực nặng hôm nay"),
        _document("d", "Kết quả xét nghiệm hoàn toàn khác."),
    ]

    groups = StableTextDeduplicator(hamming_threshold=3).group(documents)

    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]


def test_consensus_retains_single_source_proposals_for_review() -> None:
    document = _document("doc", "khó thở")
    first = _proposal(document, annotation_id="a1", labeler_id="model-a")
    second = _proposal(document, annotation_id="a2", labeler_id="model-b")
    conflict = _proposal(
        document,
        annotation_id="a3",
        labeler_id="model-c",
        entity_type="DISEASE",
    )
    labeler = ConsensusProposalLabeler(
        [FixtureLabeler([first]), FixtureLabeler([second]), FixtureLabeler([conflict])],
        min_votes=2,
    )

    proposals = list(labeler.propose([document]))

    symptom = next(value for value in proposals if value.entity_type == "SYMPTOM")
    disease = next(value for value in proposals if value.entity_type == "DISEASE")
    assert symptom.layer is AnnotationLayer.SILVER
    assert symptom.metadata["vote_count"] == "2"
    assert disease.review_status is ReviewStatus.NEEDS_REVIEW


def test_coverage_rank_uses_gap_disagreement_and_source_quality() -> None:
    common = _document("common", "đau", source_quality="0.2")
    rare = _document(
        "rare",
        "ảo giác",
        source_quality="1.0",
        relation_density="1.0",
    )
    annotations = [
        _proposal(common, annotation_id="common-1", labeler_id="rules"),
        replace(
            _proposal(
                rare,
                annotation_id="rare-1",
                labeler_id="consensus",
                review_status=ReviewStatus.NEEDS_REVIEW,
            ),
            metadata={"vote_count": "1"},
        ),
    ]
    planner = CoverageCubePlanner(
        [CoverageTarget((("entity_type", "SYMPTOM"),), target=20)]
    )

    report = planner.report("snapshot", [common, rare], annotations)
    ranked = planner.priorities([common, rare], annotations)

    assert report.cells[0].observed == 2
    assert report.cells[0].gap_ratio == 0.9
    assert ranked[0].document_id == "rare"
    assert ranked[0].score > ranked[1].score


def test_review_jsonl_round_trip_preserves_provenance() -> None:
    document = _document("doc", "sốt")
    proposal = replace(
        _proposal(document, annotation_id="a1", labeler_id="model-a"),
        review_status=ReviewStatus.ACCEPTED,
        layer=AnnotationLayer.GOLD,
        metadata={"reviewer": "clinical-reviewer"},
    )
    backend = JsonlReviewBackend()

    payload = backend.export([document], [proposal])
    imported = backend.import_reviewed(payload)

    assert imported == (proposal,)
    assert backend.export([document], [proposal]) == payload


def test_hosted_labeler_rejects_entire_batch_before_delegate_call() -> None:
    open_document = _document("open", "sốt")
    private_document = MinedDocument(
        document_id="private",
        text="khó thở",
        language="vi",
        note_type="discharge_note",
        source_artifact_id="private:artifact",
        access_class=AccessClass.DUA,
        redistribution=RedistributionPolicy.PROHIBITED,
        hosted_processing_allowed=False,
    )
    hosted = PolicyAwareProposalLabelerAdapter(
        FixtureLabeler([]),
        allow_document=lambda document: document.hosted_processing_allowed,
    )

    with pytest.raises(PermissionError, match="private"):
        list(hosted.propose([open_document, private_document]))
