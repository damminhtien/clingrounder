"""Fail-closed audit for files included in a public Git release.

Mining release locks answer whether an experiment can be reconstructed from the same
bytes. This module answers a different question: whether each tracked byte is allowed
in a public repository. Restricted bytes remain in local or content-addressed storage;
Git retains only policy, checksums, provenance, and reconstruction instructions.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
from collections import Counter
from collections.abc import Sequence
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medical_kg_nlp.utils.hashing import sha256_file

__all__ = [
    "PublicationDisposition",
    "LocalArtifactInventory",
    "LocalArtifactRecord",
    "PublicPathRule",
    "PublicRepositoryPolicy",
    "PublicReleaseIssue",
    "PublicReleaseReport",
    "audit_public_repository",
    "build_local_artifact_inventory",
    "load_public_repository_policy",
    "report_json",
]

_TEXT_SCAN_LIMIT_BYTES = 2 * 1024 * 1024
_SECRET_PATTERNS = (
    ("huggingface_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    (
        "credential_assignment",
        re.compile(
            r"(?i)(?:VAST_API_KEY|HF_TOKEN|OPENAI_API_KEY|AWS_SECRET_ACCESS_KEY)"
            r"\s*[:=]\s*[\"']?(?!<|\$|\{)[A-Za-z0-9_+/=-]{20,}"
        ),
    ),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
)


class PublicationDisposition(str, Enum):
    """Whether repository bytes may be included in a public Git tree."""

    REDISTRIBUTABLE = "redistributable"
    REDISTRIBUTABLE_WITH_NOTICE = "redistributable_with_notice"
    MANIFEST_ONLY = "manifest_only"
    LOCAL_ONLY = "local_only"


class PublicPathRule(BaseModel):
    """One ordered path classification in a public repository policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    patterns: tuple[str, ...] = Field(min_length=1)
    disposition: PublicationDisposition
    rationale: str = Field(min_length=1)
    source_ids: tuple[str, ...] = ()
    notice_path: str | None = None
    allow_large_files: bool = False
    inventory_local_bytes: bool = False

    @field_validator("patterns")
    @classmethod
    def validate_patterns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_portable_pattern(value) for value in values)

    @field_validator("notice_path")
    @classmethod
    def validate_notice_path(cls, value: str | None) -> str | None:
        return None if value is None else _portable_path(value)

    @model_validator(mode="after")
    def validate_notice_contract(self) -> "PublicPathRule":
        requires_notice = (
            self.disposition is PublicationDisposition.REDISTRIBUTABLE_WITH_NOTICE
        )
        if requires_notice != (self.notice_path is not None):
            raise ValueError(
                "redistributable_with_notice rules require notice_path, and other rules must omit it"
            )
        return self


class PublicRepositoryPolicy(BaseModel):
    """Checked-in, fail-closed policy for public repository contents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["medical-kg.public-repository-policy.v1"]
    protected_roots: tuple[str, ...] = Field(min_length=1)
    max_tracked_file_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    source_registry_path: str
    rules: tuple[PublicPathRule, ...] = Field(min_length=1)
    required_tracked_paths: tuple[str, ...] = ()
    secret_scan_excludes: tuple[str, ...] = ()

    @field_validator("protected_roots", "required_tracked_paths")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_portable_path(value).rstrip("/") for value in values)

    @field_validator("source_registry_path")
    @classmethod
    def validate_source_registry_path(cls, value: str) -> str:
        return _portable_path(value)

    @field_validator("secret_scan_excludes")
    @classmethod
    def validate_secret_excludes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_portable_pattern(value) for value in values)

    @model_validator(mode="after")
    def validate_unique_rule_ids(self) -> "PublicRepositoryPolicy":
        identifiers = [rule.id for rule in self.rules]
        duplicates = sorted(
            {value for value in identifiers if identifiers.count(value) > 1}
        )
        if duplicates:
            raise ValueError(f"duplicate public path rule IDs: {', '.join(duplicates)}")
        return self


class PublicReleaseIssue(BaseModel):
    """One publication blocker without exposing matched secret contents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    path: str
    message: str
    rule_id: str | None = None
    line: int | None = None


class PublicReleaseReport(BaseModel):
    """Deterministic report returned by the public release audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["medical-kg.public-release-report.v1"]
    valid: bool
    policy_path: str
    tracked_file_count: int
    protected_file_count: int
    disposition_counts: dict[str, int]
    issues: tuple[PublicReleaseIssue, ...]


class LocalArtifactRecord(BaseModel):
    """Content identity for one non-public file retained outside Git."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    rule_id: str
    disposition: Literal["manifest_only", "local_only"]
    source_ids: tuple[str, ...]
    byte_size: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


class LocalArtifactInventory(BaseModel):
    """Deterministic inventory that preserves provenance without restricted bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["medical-kg.local-artifact-inventory.v1"]
    policy_path: str
    policy_sha256: str = Field(min_length=64, max_length=64)
    artifact_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    artifacts: tuple[LocalArtifactRecord, ...]


def load_public_repository_policy(path: str | Path) -> PublicRepositoryPolicy:
    """Load a strict publication policy without accepting implicit keys."""

    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: public repository policy must be a mapping")
    return PublicRepositoryPolicy.model_validate(raw)


def audit_public_repository(
    root: str | Path,
    policy_path: str | Path,
    *,
    tracked_paths: Sequence[str] | None = None,
) -> PublicReleaseReport:
    """Audit tracked paths, large files, notices, and common credential shapes.

    ``tracked_paths`` is injectable for unit tests. Production callers omit it so Git is
    the source of truth for what a public clone would receive.
    """

    repository_root = Path(root).resolve()
    policy_source = Path(policy_path).resolve()
    policy = load_public_repository_policy(policy_source)
    tracked = tuple(
        sorted(
            _portable_path(path)
            for path in (
                tracked_paths
                if tracked_paths is not None
                else _git_tracked_paths(repository_root)
            )
        )
    )
    tracked_set = set(tracked)
    issues: list[PublicReleaseIssue] = []
    counts: Counter[str] = Counter()
    protected_count = 0

    from medical_kg_nlp.mining.registry import load_source_registry

    source_registry = load_source_registry(repository_root / policy.source_registry_path)
    known_source_ids = {source.id for source in source_registry.resources}
    for policy_rule in policy.rules:
        for source_id in policy_rule.source_ids:
            if source_id not in known_source_ids:
                issues.append(
                    PublicReleaseIssue(
                        code="unknown_source_id",
                        path=policy.source_registry_path,
                        rule_id=policy_rule.id,
                        message=f"Publication rule references unknown source ID: {source_id}",
                    )
                )

    for required_path in policy.required_tracked_paths:
        if required_path not in tracked_set:
            issues.append(
                PublicReleaseIssue(
                    code="missing_required_provenance",
                    path=required_path,
                    message="Required public provenance file is not tracked.",
                )
            )

    for relative_path in tracked:
        file_path = repository_root / relative_path
        path_rule = _matching_rule(relative_path, policy.rules)
        if _is_protected(relative_path, policy.protected_roots):
            protected_count += 1
            if path_rule is None:
                issues.append(
                    PublicReleaseIssue(
                        code="unclassified_protected_path",
                        path=relative_path,
                        message="Protected path has no explicit publication rule.",
                    )
                )
        if path_rule is not None:
            counts[path_rule.disposition.value] += 1
            if path_rule.disposition in {
                PublicationDisposition.LOCAL_ONLY,
                PublicationDisposition.MANIFEST_ONLY,
            }:
                issues.append(
                    PublicReleaseIssue(
                        code="restricted_path_tracked",
                        path=relative_path,
                        rule_id=path_rule.id,
                        message=(
                            f"Path is classified as {path_rule.disposition.value}; retain its bytes "
                            "outside Git and publish only approved provenance."
                        ),
                    )
                )
            elif (
                path_rule.notice_path is not None
                and path_rule.notice_path not in tracked_set
            ):
                issues.append(
                    PublicReleaseIssue(
                        code="missing_attribution_notice",
                        path=relative_path,
                        rule_id=path_rule.id,
                        message=f"Required notice is not tracked: {path_rule.notice_path}",
                    )
                )
            if (
                path_rule.disposition
                not in {
                    PublicationDisposition.LOCAL_ONLY,
                    PublicationDisposition.MANIFEST_ONLY,
                }
                and file_path.is_file()
                and file_path.stat().st_size > policy.max_tracked_file_bytes
                and not path_rule.allow_large_files
            ):
                issues.append(
                    PublicReleaseIssue(
                        code="oversized_tracked_file",
                        path=relative_path,
                        rule_id=path_rule.id,
                        message=(
                            f"Tracked file exceeds {policy.max_tracked_file_bytes} bytes; "
                            "publish a manifest and acquisition path instead."
                        ),
                    )
                )

        if not _matches_any(relative_path, policy.secret_scan_excludes):
            issues.extend(_scan_file_for_secrets(file_path, relative_path))

    ordered_issues = tuple(
        sorted(issues, key=lambda issue: (issue.path, issue.code, issue.line or 0))
    )
    return PublicReleaseReport(
        schema_version="medical-kg.public-release-report.v1",
        valid=not ordered_issues,
        policy_path=_display_path(policy_source, repository_root),
        tracked_file_count=len(tracked),
        protected_file_count=protected_count,
        disposition_counts=dict(sorted(counts.items())),
        issues=ordered_issues,
    )


def build_local_artifact_inventory(
    root: str | Path,
    policy_path: str | Path,
) -> LocalArtifactInventory:
    """Fingerprint explicitly inventoried local-only and manifest-only files.

    SCALING: only roots derived from rules with ``inventory_local_bytes=true`` are
    traversed. Generated run directories remain local but do not become an accidental,
    ever-growing release contract.
    """

    repository_root = Path(root).resolve()
    policy_source = Path(policy_path).resolve()
    policy = load_public_repository_policy(policy_source)
    candidates: set[Path] = set()
    for inventory_rule in policy.rules:
        if not inventory_rule.inventory_local_bytes:
            continue
        if inventory_rule.disposition not in {
            PublicationDisposition.LOCAL_ONLY,
            PublicationDisposition.MANIFEST_ONLY,
        }:
            raise ValueError(
                f"Rule {inventory_rule.id} inventories bytes that are already redistributable"
            )
        for pattern in inventory_rule.patterns:
            search_root = repository_root / _static_pattern_prefix(pattern)
            if search_root.is_file():
                candidates.add(search_root)
            elif search_root.is_dir():
                candidates.update(path for path in search_root.rglob("*") if path.is_file())

    records: list[LocalArtifactRecord] = []
    for path in sorted(candidates):
        relative_path = path.relative_to(repository_root).as_posix()
        matched_rule = _matching_rule(relative_path, policy.rules)
        if matched_rule is None or not matched_rule.inventory_local_bytes:
            continue
        if matched_rule.disposition not in {
            PublicationDisposition.LOCAL_ONLY,
            PublicationDisposition.MANIFEST_ONLY,
        }:
            continue
        disposition: Literal["manifest_only", "local_only"] = (
            "manifest_only"
            if matched_rule.disposition is PublicationDisposition.MANIFEST_ONLY
            else "local_only"
        )
        records.append(
            LocalArtifactRecord(
                path=relative_path,
                rule_id=matched_rule.id,
                disposition=disposition,
                source_ids=matched_rule.source_ids,
                byte_size=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return LocalArtifactInventory(
        schema_version="medical-kg.local-artifact-inventory.v1",
        policy_path=_display_path(policy_source, repository_root),
        policy_sha256=sha256_file(policy_source),
        artifact_count=len(records),
        total_bytes=sum(record.byte_size for record in records),
        artifacts=tuple(records),
    )


def _git_tracked_paths(root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return tuple(
        value.decode("utf-8")
        for value in completed.stdout.split(b"\0")
        if value
    )


def _matching_rule(path: str, rules: Sequence[PublicPathRule]) -> PublicPathRule | None:
    # INVARIANT: YAML order is the explicit precedence contract; broad fallbacks belong last.
    return next(
        (rule for rule in rules if _matches_any(path, rule.patterns)),
        None,
    )


def _is_protected(path: str, roots: Sequence[str]) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in roots)


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _scan_file_for_secrets(path: Path, display_path: str) -> list[PublicReleaseIssue]:
    if not path.is_file() or path.stat().st_size > _TEXT_SCAN_LIMIT_BYTES:
        return []
    payload = path.read_bytes()
    if b"\0" in payload:
        return []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return []
    issues: list[PublicReleaseIssue] = []
    for secret_type, pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            # PRIVACY: report only the pattern class and location, never credential bytes.
            issues.append(
                PublicReleaseIssue(
                    code="potential_secret",
                    path=display_path,
                    line=text.count("\n", 0, match.start()) + 1,
                    message=f"Potential {secret_type} detected; inspect and rotate before release.",
                )
            )
    return issues


def _portable_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    if not path or path.startswith("/") or PureWindowsPath(path).is_absolute():
        raise ValueError("public release paths must be non-empty repository-relative paths")
    parts = PurePosixPath(path).parts
    if ".." in parts or "." in parts:
        raise ValueError("public release paths cannot contain traversal components")
    return PurePosixPath(path).as_posix()


def _portable_pattern(value: str) -> str:
    pattern = _portable_path(value)
    if pattern.endswith("/"):
        raise ValueError("public release patterns must name files or use an explicit wildcard")
    return pattern


def _static_pattern_prefix(pattern: str) -> Path:
    parts: list[str] = []
    for part in PurePosixPath(pattern).parts:
        if any(character in part for character in "*?["):
            break
        parts.append(part)
    if not parts:
        raise ValueError(f"inventory pattern requires a static repository prefix: {pattern}")
    return Path(*parts)


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def report_json(report: PublicReleaseReport) -> str:
    """Serialize a report consistently for CLI output and archived evidence."""

    return json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
