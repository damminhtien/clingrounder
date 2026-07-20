"""Tests for fail-closed crosswalk concept-link materialization."""

from __future__ import annotations

from dataclasses import replace

from medical_kg_nlp.mining.crosswalk_links import (
    CrosswalkLinkMaterializationPolicy,
    load_crosswalk_link_policy,
    materialize_exact_crosswalk_links,
)
from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    ReviewStatus,
)
from medical_kg_nlp.schema.types import CodeSystem


def _annotation(*, concepts: tuple[ConceptLink, ...] = ()) -> AnnotationProposal:
    return AnnotationProposal(
        annotation_id="annotation-1",
        document_id="document-1",
        span=(0, 13),
        text="Wilson disease",
        entity_type="DISEASE",
        assertions=(),
        concepts=concepts,
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="source_structured_annotation",
        labeler_id="clinicaltrials-fixture",
        review_status=ReviewStatus.PROPOSED,
        source_label="CTGOV_CONDITION",
        metadata={"source_field": "condition"},
    )


def _policy(
    *, append_non_conflicting_code_systems: bool = False
) -> CrosswalkLinkMaterializationPolicy:
    return CrosswalkLinkMaterializationPolicy(
        policy_id="clinicaltrials-mondo-fixture",
        accepted_crosswalk_policy_ids=frozenset(
            {"clinicaltrials-condition-to-mondo-v1"}
        ),
        accepted_candidate_sources=frozenset({"mondo:2026-07-06"}),
        accepted_code_systems=frozenset({CodeSystem.MONDO}),
        accepted_promotion_statuses=frozenset({"review_required"}),
        append_non_conflicting_code_systems=append_non_conflicting_code_systems,
    )


def _row(*, status: str = "unique_concept_exact") -> dict[str, object]:
    return {
        "automatic_promotion_allowed": False,
        "candidate_count": 1,
        "candidates": [
            {
                "code": "0010200",
                "code_system": "MONDO",
                "semantic_type": "DISEASE",
                "source": "mondo:2026-07-06",
            }
        ],
        "code_count": 1,
        "match_mode": "normalized_exact",
        "normalized_mention": "wilson disease",
        "policy_id": "clinicaltrials-condition-to-mondo-v1",
        "promotion_status": "review_required",
        "query_truncated": False,
        "source_entity_type": "DISEASE",
        "source_label": "CTGOV_CONDITION",
        "status": status,
    }


def test_exact_unique_crosswalk_attaches_link_without_changing_source_fields() -> None:
    annotation = _annotation()

    result = materialize_exact_crosswalk_links([annotation], [_row()], _policy())

    linked = result.annotations[0]
    assert replace(linked, concepts=(), metadata=annotation.metadata) == annotation
    assert linked.concepts == (
        ConceptLink(
            code_system="MONDO",
            code="0010200",
            terminology_version="mondo:2026-07-06",
        ),
    )
    assert linked.metadata["crosswalk_promotion_status"] == "review_required"
    assert result.report["annotation_decision_counts"] == {"linked": 1}


def test_ambiguous_crosswalk_and_existing_link_are_never_overwritten() -> None:
    ambiguous = materialize_exact_crosswalk_links(
        [_annotation()], [_row(status="ambiguous_code_exact")], _policy()
    )
    existing = ConceptLink("MONDO", "9999999", "manual-review:v1")
    conflicting = materialize_exact_crosswalk_links(
        [_annotation(concepts=(existing,))], [_row()], _policy()
    )

    assert ambiguous.annotations[0].concepts == ()
    assert ambiguous.report["crosswalk_row_decision_counts"] == {
        "status:ambiguous_code_exact": 1
    }
    assert conflicting.annotations[0].concepts == (existing,)
    assert conflicting.report["annotation_decision_counts"] == {
        "existing_code_system_conflict": 1
    }


def test_policy_can_append_new_code_system_without_overwriting_existing_link() -> None:
    icd_link = ConceptLink("ICD-10", "E83.01", "tt06:2026")

    result = materialize_exact_crosswalk_links(
        [_annotation(concepts=(icd_link,))],
        [_row()],
        _policy(append_non_conflicting_code_systems=True),
    )

    assert result.annotations[0].concepts == (
        icd_link,
        ConceptLink("MONDO", "0010200", "mondo:2026-07-06"),
    )
    assert result.report["annotation_decision_counts"] == {
        "linked_additional_code_system": 1
    }


def test_checked_in_clinicaltrials_policy_pins_mondo_release() -> None:
    policy = load_crosswalk_link_policy(
        "configs/mining/linking/clinicaltrials-mondo.yaml"
    )

    assert policy.accepted_candidate_sources == frozenset({"mondo:2026-07-06"})
    assert policy.accepted_code_systems == frozenset({CodeSystem.MONDO})
    assert policy.append_non_conflicting_code_systems is False


def test_checked_in_pmc_policy_allows_only_cross_system_append() -> None:
    policy = load_crosswalk_link_policy(
        "configs/mining/linking/pmc-case-mondo-hpo.yaml"
    )

    assert policy.accepted_candidate_sources == frozenset(
        {"mondo:2026-07-06", "hpo:2026-06-23"}
    )
    assert policy.accepted_code_systems == frozenset(
        {CodeSystem.MONDO, CodeSystem.HPO}
    )
    assert policy.append_non_conflicting_code_systems is True
