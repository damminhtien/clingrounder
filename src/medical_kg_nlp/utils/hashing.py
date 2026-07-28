"""Stable content fingerprints shared by data and runtime artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["sha256_directory", "sha256_file", "sha256_text"]


def sha256_directory(path: str | Path) -> str:
    """Hash relative file names and contents for one immutable local artifact.

    SCALING: model directories are streamed file by file so large adapter weights do not need
    to fit in memory. Including relative paths prevents two differently structured artifacts
    with the same concatenated bytes from sharing an identity.
    """

    root = Path(path)
    files = sorted(value for value in root.rglob("*") if value.is_file())
    if not files:
        raise ValueError(f"Artifact directory {root} is missing or empty")
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading large mining artifacts into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
