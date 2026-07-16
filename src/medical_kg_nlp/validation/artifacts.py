"""Release-only checks for expected files and deterministic ZIP structure."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from medical_kg_nlp.validation.profiles import ValidationProfile, ValidationSeverity

__all__ = ["ArtifactValidationIssue", "validate_artifact"]

_DETERMINISTIC_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ArtifactValidationIssue:
    """A release artifact structure or determinism failure."""

    kind: str
    path: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR

    def to_json(self) -> dict[str, str]:
        """Return a flat machine-readable issue payload."""

        return {
            "kind": self.kind,
            "path": self.path,
            "message": self.message,
            "severity": self.severity.value,
        }


def validate_artifact(
    path: str | Path,
    *,
    profile: ValidationProfile,
    expected_files: tuple[str, ...] = (),
) -> list[ArtifactValidationIssue]:
    """Validate directory members, or ZIP members and deterministic metadata, for release."""

    if profile is not ValidationProfile.RELEASE:
        return []
    artifact = Path(path)
    if artifact.is_dir():
        actual = tuple(sorted(item.name for item in artifact.iterdir() if item.is_file()))
        return _expected_file_issues(actual, expected_files, str(artifact))
    if artifact.suffix.casefold() != ".zip":
        return [
            ArtifactValidationIssue(
                "unsupported_artifact",
                str(artifact),
                "Release artifact must be a directory or ZIP file.",
            )
        ]
    try:
        with zipfile.ZipFile(artifact) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
    except (OSError, zipfile.BadZipFile) as error:
        return [ArtifactValidationIssue("invalid_zip", str(artifact), str(error))]

    names = tuple(info.filename for info in infos)
    issues = _expected_file_issues(names, expected_files, str(artifact))
    if names != tuple(sorted(names)):
        issues.append(
            ArtifactValidationIssue(
                "nondeterministic_zip_order",
                str(artifact),
                "ZIP members must be written in lexical order.",
            )
        )
    for info in infos:
        if Path(info.filename).name != info.filename:
            issues.append(
                ArtifactValidationIssue(
                    "nested_zip_member",
                    info.filename,
                    "ZIP members must be flat files.",
                )
            )
        if info.date_time != _DETERMINISTIC_ZIP_TIMESTAMP:
            issues.append(
                ArtifactValidationIssue(
                    "nondeterministic_zip_timestamp",
                    info.filename,
                    "ZIP members must use the fixed 1980-01-01 timestamp.",
                )
            )
    return issues


def _expected_file_issues(
    actual: tuple[str, ...],
    expected: tuple[str, ...],
    path: str,
) -> list[ArtifactValidationIssue]:
    if not expected:
        return []
    actual_set = set(actual)
    expected_set = set(expected)
    issues = [
        ArtifactValidationIssue(
            "missing_expected_file",
            path,
            f"Missing expected file {name!r}.",
        )
        for name in sorted(expected_set - actual_set)
    ]
    issues.extend(
        ArtifactValidationIssue(
            "unexpected_file",
            path,
            f"Unexpected file {name!r}.",
        )
        for name in sorted(actual_set - expected_set)
    )
    return issues
