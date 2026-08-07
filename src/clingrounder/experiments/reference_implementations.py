"""Pinned external-reference registry and reproducible checkout verification.

The checkouts are deliberately ignored by Git. This module tracks only the small,
reviewable facts needed to reproduce an architecture audit on another machine.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ReferenceImplementation",
    "ReferenceRegistry",
    "ReferenceVerification",
    "load_reference_registry",
    "sync_reference_checkouts",
    "verify_reference_checkouts",
]

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_LICENSE_STATUSES = frozenset(
    {"verified", "partial", "file_level", "user_authorized", "unverified"}
)


@dataclass(frozen=True, slots=True)
class ReferenceImplementation:
    """One immutable source used for architecture study, not a runtime dependency."""

    source_id: str
    repository_url: str
    revision: str
    checkout: str
    license_status: str
    license_spdx: str | None
    license_evidence: tuple[str, ...]
    inspected_paths: tuple[str, ...]
    adopt: tuple[str, ...]
    reject: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReferenceRegistry:
    """Validated reference-source registry."""

    schema_version: str
    sources: tuple[ReferenceImplementation, ...]


@dataclass(frozen=True, slots=True)
class ReferenceVerification:
    """Local verification result for one pinned checkout."""

    source_id: str
    expected_revision: str
    actual_revision: str | None
    checkout_exists: bool
    revision_matches: bool
    missing_inspected_paths: tuple[str, ...]
    missing_license_evidence: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return (
            self.checkout_exists
            and self.revision_matches
            and not self.missing_inspected_paths
            and not self.missing_license_evidence
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "expected_revision": self.expected_revision,
            "actual_revision": self.actual_revision,
            "checkout_exists": self.checkout_exists,
            "revision_matches": self.revision_matches,
            "missing_inspected_paths": list(self.missing_inspected_paths),
            "missing_license_evidence": list(self.missing_license_evidence),
            "valid": self.valid,
        }


def load_reference_registry(path: str | Path) -> ReferenceRegistry:
    """Load and validate the tracked reference registry."""

    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{registry_path}: expected a JSON object.")
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != "clinical-nlp-reference-registry.v1":
        raise ValueError(f"{registry_path}: unsupported schema_version {schema_version!r}.")
    rows = payload.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{registry_path}: sources must be a non-empty list.")

    sources = tuple(_source_from_row(row, registry_path) for row in rows)
    source_ids = [source.source_id for source in sources]
    checkouts = [source.checkout for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{registry_path}: source_id values must be unique.")
    if len(checkouts) != len(set(checkouts)):
        raise ValueError(f"{registry_path}: checkout values must be unique.")
    return ReferenceRegistry(schema_version=schema_version, sources=sources)


def verify_reference_checkouts(
    registry: ReferenceRegistry,
    checkout_root: str | Path,
) -> tuple[ReferenceVerification, ...]:
    """Verify revisions and reviewed paths without importing third-party code."""

    root = Path(checkout_root)
    results: list[ReferenceVerification] = []
    for source in registry.sources:
        checkout = root / source.checkout
        exists = checkout.is_dir() and (checkout / ".git").exists()
        actual_revision = _git_revision(checkout) if exists else None
        missing_inspected = tuple(
            path for path in source.inspected_paths if not (checkout / path).exists()
        )
        missing_license = tuple(
            path for path in source.license_evidence if not (checkout / path).exists()
        )
        results.append(
            ReferenceVerification(
                source_id=source.source_id,
                expected_revision=source.revision,
                actual_revision=actual_revision,
                checkout_exists=exists,
                revision_matches=actual_revision == source.revision,
                missing_inspected_paths=missing_inspected,
                missing_license_evidence=missing_license,
            )
        )
    return tuple(results)


def sync_reference_checkouts(
    registry: ReferenceRegistry,
    checkout_root: str | Path,
) -> tuple[ReferenceVerification, ...]:
    """Create or update detached checkouts at the exact reviewed revisions.

    SCALING: Hugging Face LFS downloads are disabled for audit-only checkouts. Model
    weights should be materialized through the model cache only when a benchmark uses them.
    """

    root = Path(checkout_root)
    root.mkdir(parents=True, exist_ok=True)
    for source in registry.sources:
        checkout = root / source.checkout
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        if not (checkout / ".git").exists():
            _run_git(("clone", "--no-checkout", source.repository_url, str(checkout)), env=env)
        _run_git(
            ("-C", str(checkout), "fetch", "--depth", "1", "origin", source.revision),
            env=env,
        )
        _run_git(
            ("-C", str(checkout), "checkout", "--detach", source.revision),
            env=env,
        )
    return verify_reference_checkouts(registry, root)


def _source_from_row(row: Any, path: Path) -> ReferenceImplementation:
    if not isinstance(row, dict):
        raise ValueError(f"{path}: every source must be a JSON object.")
    source_id = _required_text(row, "source_id", path)
    revision = _required_text(row, "revision", path)
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError(f"{path}: {source_id} revision must be a full SHA-1.")
    license_status = _required_text(row, "license_status", path)
    if license_status not in _LICENSE_STATUSES:
        raise ValueError(f"{path}: {source_id} has invalid license_status.")
    license_spdx_value = row.get("license_spdx")
    license_spdx = str(license_spdx_value) if license_spdx_value is not None else None
    if license_status == "verified" and not license_spdx:
        raise ValueError(f"{path}: verified source {source_id} needs license_spdx.")
    return ReferenceImplementation(
        source_id=source_id,
        repository_url=_required_text(row, "repository_url", path),
        revision=revision,
        checkout=_required_text(row, "checkout", path),
        license_status=license_status,
        license_spdx=license_spdx,
        license_evidence=_text_tuple(row, "license_evidence", path),
        inspected_paths=_nonempty_text_tuple(row, "inspected_paths", path),
        adopt=_nonempty_text_tuple(row, "adopt", path),
        reject=_nonempty_text_tuple(row, "reject", path),
    )


def _required_text(row: dict[str, Any], key: str, path: Path) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"{path}: {key} must be non-empty.")
    return value


def _text_tuple(row: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    values = row.get(key, [])
    if not isinstance(values, list):
        raise ValueError(f"{path}: {key} must be a list.")
    result = tuple(str(value).strip() for value in values if str(value).strip())
    if len(result) != len(values):
        raise ValueError(f"{path}: {key} contains an empty value.")
    return result


def _nonempty_text_tuple(
    row: dict[str, Any],
    key: str,
    path: Path,
) -> tuple[str, ...]:
    values = _text_tuple(row, key, path)
    if not values:
        raise ValueError(f"{path}: {key} must be non-empty.")
    return values


def _git_revision(checkout: Path) -> str | None:
    completed = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_git(arguments: tuple[str, ...], *, env: dict[str, str]) -> None:
    subprocess.run(("git", *arguments), check=True, env=env)
