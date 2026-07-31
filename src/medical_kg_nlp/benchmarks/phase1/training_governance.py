"""Machine-checkable Phase 1 supervision and model-promotion governance."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Phase1TrainingGovernance",
    "load_phase1_training_governance",
]


class _FrozenPolicy(BaseModel):
    """Reject undeclared policy fields so a typo cannot weaken a training contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ManualGoldPolicy(_FrozenPolicy):
    """Final-fit policy for the repository's manually reviewed corpus."""

    path: str = Field(min_length=1)
    source_documents: str = Field(min_length=1)
    expected_document_count: Literal[100]
    usage: Literal["train_all"]
    legacy_split_role: Literal["diagnostic_only"]


class AuthorizedGroundTruthPolicy(_FrozenPolicy):
    """Pinned private ground truth that is authorized for supervised training."""

    source_id: Literal["phase1_part2_leaked_bundle"]
    archive_env: str = Field(min_length=1)
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_zip_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gt_zip_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_document_count: Literal[100]
    usage: Literal["supervised_training"]
    offset_coordinate_view: Literal["crlf_to_lf_child_document"]


class TeacherReferencePolicy(_FrozenPolicy):
    """External predictions may teach a model but cannot enter final inference directly."""

    source_id: Literal["friend31"]
    role: Literal["distillation_reference_only"]
    runtime_source_allowed: Literal[False]
    submission_seed_allowed: Literal[False]


class DecisionAuthorityPolicy(_FrozenPolicy):
    """Separate diagnostic telemetry from official model-quality decisions."""

    local_metrics: Literal["diagnostic_only"]
    local_can_promote: Literal[False]
    local_can_reject: Literal[False]
    official_submission: Literal["sole_promotion_and_rejection_authority"]
    hard_validation_can_block_packaging: Literal[True]
    major_change_requires_submission_artifact: Literal[True]
    major_change_may_close_without_artifact: Literal[False]


class ReproducibilityPolicy(_FrozenPolicy):
    """Requirements that prevent a winning artifact from depending on an external output."""

    require_repository_owned_inference: Literal[True]
    require_pinned_checkpoint: Literal[True]
    require_pinned_config: Literal[True]
    require_source_fingerprints: Literal[True]


class Phase1TrainingGovernance(_FrozenPolicy):
    """Complete policy applied to final-fit Phase 1 experiments."""

    schema_version: Literal["phase1-training-governance.v1"]
    effective_from: str = Field(min_length=1)
    manual_gold: ManualGoldPolicy
    authorized_ground_truth: AuthorizedGroundTruthPolicy
    friend31: TeacherReferencePolicy
    decision_authority: DecisionAuthorityPolicy
    reproducibility: ReproducibilityPolicy

    def can_local_metric_decide(self) -> bool:
        """Return false by contract; callers must record an official submission instead."""

        return self.decision_authority.local_can_promote or (
            self.decision_authority.local_can_reject
        )

    def is_runtime_source_allowed(self, source_id: str) -> bool:
        """Reject Friend31 while allowing repository-owned model sources."""

        if source_id == self.friend31.source_id:
            return self.friend31.runtime_source_allowed
        return True


def load_phase1_training_governance(
    path: str | Path,
) -> Phase1TrainingGovernance:
    """Load the strict YAML policy used by final-fit dataset and submission tooling."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return Phase1TrainingGovernance.model_validate(payload)
