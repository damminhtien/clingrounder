"""Severity policies layered over invariant-oriented prediction issues."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from medical_kg_nlp.schema.validator import PredictionValidationIssue

__all__ = [
    "ProfiledValidationIssue",
    "ValidationProfile",
    "ValidationSeverity",
    "apply_validation_profile",
]

_HARD_ERROR_KINDS = frozenset(
    {
        "schema",
        "duplicate_document_id",
        "duplicate_entity_id",
        "duplicate_relation_id",
        "offset",
        "invalid_code_system",
        "invalid_candidate_code_system",
        "invalid_relation",
        "invalid_evidence_span",
    }
)


class ValidationProfile(str, Enum):
    """Choose validation scope without weakening core invariants."""

    CORE = "core"
    DEVELOPMENT = "development"
    RELEASE = "release"


class ValidationSeverity(str, Enum):
    """Machine-readable issue disposition used by CLIs and reports."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ProfiledValidationIssue:
    """A detected issue plus its profile-specific severity."""

    issue: PredictionValidationIssue
    severity: ValidationSeverity

    def to_json(self) -> dict[str, str]:
        """Return a flat payload for JSONL and CLI output."""

        return {**self.issue.to_json(), "severity": self.severity.value}


def apply_validation_profile(
    issues: list[PredictionValidationIssue],
    profile: ValidationProfile,
    *,
    terminology_loaded: bool,
) -> list[ProfiledValidationIssue]:
    """Classify issues while keeping schema, offset, type, and relation errors blocking."""

    return [
        ProfiledValidationIssue(
            issue=issue,
            severity=_severity(issue, profile, terminology_loaded=terminology_loaded),
        )
        for issue in issues
    ]


def _severity(
    issue: PredictionValidationIssue,
    profile: ValidationProfile,
    *,
    terminology_loaded: bool,
) -> ValidationSeverity:
    if issue.kind in _HARD_ERROR_KINDS:
        return ValidationSeverity.ERROR
    if issue.kind == "text_hash_mismatch":
        return (
            ValidationSeverity.ERROR
            if profile is ValidationProfile.RELEASE
            else ValidationSeverity.WARNING
        )
    if issue.kind == "unknown_dictionary_code":
        is_candidate = ".candidates[" in issue.path
        if terminology_loaded and not is_candidate:
            # INVARIANT: an assigned output code must exist in the loaded terminology.
            return ValidationSeverity.ERROR
        if profile is ValidationProfile.RELEASE:
            return ValidationSeverity.ERROR
        return ValidationSeverity.WARNING
    return (
        ValidationSeverity.ERROR
        if profile is ValidationProfile.RELEASE
        else ValidationSeverity.WARNING
    )
