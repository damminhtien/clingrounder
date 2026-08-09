"""Tests for the portable public-review agreement artifact."""

from __future__ import annotations

import json

import pytest

from clingrounder.evaluation.review_agreement import ReviewAgreementArtifact


def _report() -> dict[str, object]:
    return {
        "reviewed_document_count": 10,
        "double_reviewed_document_count": 2,
        "double_review_fraction": 0.2,
        "span_type_agreement": 0.95,
        "assertion_agreement": 0.9,
        "relation_agreement": None,
    }


def test_review_agreement_artifact_is_text_free_and_deterministic(tmp_path) -> None:
    artifact = ReviewAgreementArtifact.from_report("dataset", "1.0.0", _report())
    output = tmp_path / "review" / "agreement.json"

    artifact.write(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "clingrounder.review-agreement.v1"
    assert payload["dataset_id"] == "dataset"
    assert "reviewer" not in output.read_text(encoding="utf-8").casefold()
    assert "clinical" not in output.read_text(encoding="utf-8").casefold()


def test_review_agreement_artifact_rejects_inconsistent_fraction() -> None:
    report = {**_report(), "double_review_fraction": 0.3}

    with pytest.raises(ValueError, match="does not match reviewer counts"):
        ReviewAgreementArtifact.from_report("dataset", "1.0.0", report)
