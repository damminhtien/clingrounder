"""Validation severity and release-artifact regression tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

from medical_kg_nlp.schema.validator import PredictionValidationIssue
from medical_kg_nlp.validation import (
    ValidationProfile,
    ValidationSeverity,
    apply_validation_profile,
    validate_artifact,
)


def test_hard_invariants_are_errors_in_every_profile() -> None:
    issue = PredictionValidationIssue("offset", "$.entities[0]", "mismatch")

    severities = {
        apply_validation_profile([issue], profile, terminology_loaded=False)[0].severity
        for profile in ValidationProfile
    }

    assert severities == {ValidationSeverity.ERROR}


def test_development_warns_for_hash_and_unknown_internal_candidate() -> None:
    issues = [
        PredictionValidationIssue("text_hash_mismatch", "$.text_hash", "mismatch"),
        PredictionValidationIssue(
            "unknown_dictionary_code",
            "$.entities[0].candidates[0].code",
            "unknown",
        ),
        PredictionValidationIssue(
            "unknown_dictionary_code",
            "$.entities[0].code",
            "unknown",
        ),
    ]

    development = apply_validation_profile(
        issues,
        ValidationProfile.DEVELOPMENT,
        terminology_loaded=True,
    )
    release = apply_validation_profile(
        issues,
        ValidationProfile.RELEASE,
        terminology_loaded=True,
    )

    assert [item.severity for item in development] == [
        ValidationSeverity.WARNING,
        ValidationSeverity.WARNING,
        ValidationSeverity.ERROR,
    ]
    assert {item.severity for item in release} == {ValidationSeverity.ERROR}


def test_artifact_checks_run_only_for_release(tmp_path: Path) -> None:
    archive = tmp_path / "output.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("2.json", "[]")
        handle.writestr("1.json", "[]")

    assert (
        validate_artifact(
            archive,
            profile=ValidationProfile.DEVELOPMENT,
            expected_files=("1.json", "2.json"),
        )
        == []
    )
    kinds = {
        issue.kind
        for issue in validate_artifact(
            archive,
            profile=ValidationProfile.RELEASE,
            expected_files=("1.json", "2.json"),
        )
    }

    assert "nondeterministic_zip_order" in kinds
    assert "nondeterministic_zip_timestamp" in kinds
