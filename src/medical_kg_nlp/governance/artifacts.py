"""Content-addressed artifact checks and local path restrictions."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from medical_kg_nlp.utils.hashing import sha256_directory, sha256_file

__all__ = [
    "ArtifactVerificationError",
    "fingerprint_artifact",
    "safe_artifact_path",
    "secure_temporary_path",
    "verify_artifact",
]


class ArtifactVerificationError(ValueError):
    """Raised when a local artifact is outside policy or has changed."""


def fingerprint_artifact(path: str | Path) -> str:
    """Return a deterministic SHA-256 for a file or directory artifact."""

    candidate = Path(path)
    if candidate.is_file():
        return sha256_file(candidate)
    if candidate.is_dir():
        return sha256_directory(candidate)
    raise ArtifactVerificationError(f"Artifact does not exist: {candidate}")


def verify_artifact(path: str | Path, expected_sha256: str) -> str:
    """Fingerprint an artifact and fail closed on an unexpected digest."""

    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ArtifactVerificationError("expected_sha256 must be a lowercase SHA-256 digest")
    actual = fingerprint_artifact(path)
    if actual != expected:
        raise ArtifactVerificationError(
            f"Artifact fingerprint mismatch for {Path(path)}: expected {expected}, got {actual}"
        )
    return actual


def safe_artifact_path(
    path: str | Path,
    *,
    allowed_roots: Iterable[str | Path],
    must_exist: bool = True,
) -> Path:
    """Resolve a local artifact and reject traversal or symlink escape."""

    try:
        candidate = Path(path).expanduser().resolve(strict=must_exist)
        roots = [Path(root).expanduser().resolve(strict=True) for root in allowed_roots]
    except FileNotFoundError as error:
        raise ArtifactVerificationError(f"Artifact path does not exist: {path}") from error
    if not any(candidate == root or root in candidate.parents for root in roots):
        raise ArtifactVerificationError(f"Artifact path is outside allowed roots: {candidate}")
    if must_exist and not os.access(candidate, os.R_OK):
        raise ArtifactVerificationError(f"Artifact is not readable: {candidate}")
    return candidate


@contextmanager
def secure_temporary_path(*, directory: str | Path | None = None, suffix: str = "") -> Iterator[Path]:
    """Yield a private temporary path and remove it unless the caller retains it."""

    root = None if directory is None else Path(directory)
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix="medical-kg-", suffix=suffix, dir=root)
    os.close(descriptor)
    path = Path(raw_path)
    path.chmod(0o600)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
