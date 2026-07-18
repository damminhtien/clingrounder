"""Inter-reviewer agreement metrics and release threshold tests."""

from __future__ import annotations

from dataclasses import replace

from medical_kg_nlp.mining.quality import (
    GoldAgreementGate,
    ReviewAgreementEvaluator,
)
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    RelationProposal,
    ReviewStatus,
)


def _document(document_id: str) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text="Sốt do cúm A.",
        language="vi",
        note_type="progress_note",
        source_artifact_id="fixture:artifact",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
    )


def _annotation(
    document: MinedDocument,
    *,
    annotation_id: str,
    reviewer: str,
    span: tuple[int, int],
    text: str,
    entity_type: str,
    assertions: tuple[str, ...] = (),
) -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document.document_id,
        span=span,
        text=text,
        entity_type=entity_type,
        assertions=assertions,
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.GOLD,
        label_source="human_review",
        labeler_id=reviewer,
        review_status=ReviewStatus.ACCEPTED,
    )


def _reviewer_labels(
    document: MinedDocument, reviewer: str
) -> tuple[AnnotationProposal, AnnotationProposal, RelationProposal]:
    symptom = _annotation(
        document,
        annotation_id=f"{reviewer}-symptom",
        reviewer=reviewer,
        span=(0, 3),
        text="Sốt",
        entity_type="SYMPTOM",
    )
    disease = _annotation(
        document,
        annotation_id=f"{reviewer}-disease",
        reviewer=reviewer,
        span=(7, 12),
        text="cúm A",
        entity_type="DISEASE",
    )
    relation = RelationProposal(
        relation_id=f"{reviewer}-relation",
        document_id=document.document_id,
        head_annotation_id=disease.annotation_id,
        tail_annotation_id=symptom.annotation_id,
        relation_type="CAUSES",
        confidence=1.0,
        layer=AnnotationLayer.GOLD,
        label_source="human_review",
        labeler_id=reviewer,
        review_status=ReviewStatus.ACCEPTED,
    )
    return symptom, disease, relation


def test_agreement_report_scores_independent_reviewers_by_semantics() -> None:
    document = _document("doc")
    left_symptom, left_disease, left_relation = _reviewer_labels(document, "left")
    right_symptom, right_disease, right_relation = _reviewer_labels(document, "right")

    report = ReviewAgreementEvaluator().evaluate(
        [document],
        [left_symptom, left_disease, right_symptom, right_disease],
        [left_relation, right_relation],
    )

    assert report.double_review_fraction == 1.0
    assert report.span_type_agreement == 1.0
    assert report.assertion_agreement == 1.0
    assert report.relation_agreement == 1.0
    assert GoldAgreementGate().validate(report, has_gold_relations=True) == ()


def test_agreement_gate_reports_assertion_disagreement_and_low_review_coverage() -> None:
    double_reviewed = _document("double")
    single_reviewed = _document("single")
    left, disease, _ = _reviewer_labels(double_reviewed, "left")
    right = replace(
        left,
        annotation_id="right-symptom",
        labeler_id="right",
        assertions=("NEGATED",),
    )
    single = _annotation(
        single_reviewed,
        annotation_id="single-ann",
        reviewer="left",
        span=(0, 3),
        text="Sốt",
        entity_type="SYMPTOM",
    )

    report = ReviewAgreementEvaluator().evaluate(
        [double_reviewed, single_reviewed],
        [left, disease, right, single],
    )
    issues = GoldAgreementGate().validate(report, has_gold_relations=False)

    assert report.double_review_fraction == 0.5
    assert report.span_type_agreement == 0.5
    assert report.assertion_agreement == 0.0
    assert any(issue.startswith("span_type_agreement") for issue in issues)
    assert any(issue.startswith("assertion_agreement") for issue in issues)


def test_agreement_gate_rejects_unmeasured_single_reviewer_gold() -> None:
    document = _document("doc")
    annotation = _annotation(
        document,
        annotation_id="ann",
        reviewer="only-reviewer",
        span=(0, 3),
        text="Sốt",
        entity_type="SYMPTOM",
    )

    report = ReviewAgreementEvaluator().evaluate([document], [annotation])
    issues = GoldAgreementGate().validate(report, has_gold_relations=False)

    assert "span_type_agreement:unmeasured" in issues
    assert "assertion_agreement:unmeasured" in issues
    assert any(issue.startswith("double_review_fraction") for issue in issues)
