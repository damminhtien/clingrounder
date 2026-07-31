"""Phase 1 final-fit supervision and decision-authority contract tests."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from medical_kg_nlp.benchmarks.phase1.training_governance import (
    Phase1TrainingGovernance,
    load_phase1_training_governance,
)


POLICY_PATH = "configs/models/phase1-training-governance-2026-07-30.yaml"


def test_checked_in_policy_uses_all_manual_and_authorized_gt_records() -> None:
    policy = load_phase1_training_governance(POLICY_PATH)

    assert policy.manual_gold.expected_document_count == 100
    assert policy.manual_gold.usage == "train_all"
    assert policy.authorized_ground_truth.expected_document_count == 100
    assert policy.authorized_ground_truth.usage == "supervised_training"
    assert (
        policy.authorized_ground_truth.offset_coordinate_view
        == "crlf_to_lf_child_document"
    )


def test_local_metrics_cannot_decide_and_friend31_cannot_run() -> None:
    policy = load_phase1_training_governance(POLICY_PATH)

    assert policy.can_local_metric_decide() is False
    assert policy.is_runtime_source_allowed("friend31") is False
    assert policy.is_runtime_source_allowed("repository_qwen") is True
    assert (
        policy.decision_authority.official_submission
        == "sole_promotion_and_rejection_authority"
    )
    assert policy.decision_authority.major_change_requires_submission_artifact is True
    assert policy.decision_authority.major_change_may_close_without_artifact is False


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("manual_gold", "expected_document_count", 76),
        ("manual_gold", "usage", "train_split"),
        ("friend31", "runtime_source_allowed", True),
        ("decision_authority", "local_can_reject", True),
        ("decision_authority", "major_change_may_close_without_artifact", True),
    ],
)
def test_policy_rejects_weakened_governance(
    section: str,
    field: str,
    value: object,
) -> None:
    policy = load_phase1_training_governance(POLICY_PATH)
    payload = copy.deepcopy(policy.model_dump())
    payload[section][field] = value

    with pytest.raises(ValidationError):
        Phase1TrainingGovernance.model_validate(payload)
