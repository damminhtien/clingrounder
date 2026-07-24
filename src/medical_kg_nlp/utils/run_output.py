from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "HashedRunOutput",
    "collect_git_metadata",
    "create_hashed_run_dir",
    "path_in_run",
]

_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_ENVIRONMENT_LOCK_CANDIDATES = (Path("uv.lock"), Path("pyproject.toml"))


@dataclass(frozen=True)
class HashedRunOutput:
    run_id: str
    run_dir: Path
    manifest_path: Path


def collect_git_metadata() -> dict[str, str | bool | None]:
    """Return source-control identity without failing outside a Git checkout."""

    commit, dirty, working_tree_hash = _git_metadata()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "working_tree_hash": working_tree_hash,
    }


def create_hashed_run_dir(
    root: str | Path,
    *,
    label: str,
    inputs: Sequence[str | Path] = (),
    resolved_config: Mapping[str, Any] | None = None,
    random_seed: int | None = None,
    command: Sequence[str] | None = None,
) -> HashedRunOutput:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = _slug(label)
    input_artifacts = [_input_artifact(item) for item in inputs]
    git_metadata = collect_git_metadata()
    git_commit = git_metadata["git_commit"]
    git_dirty = git_metadata["git_dirty"]
    working_tree_hash = git_metadata["working_tree_hash"]
    lock_hash = _environment_lock_hash()
    reproducibility_payload = {
        "label": label,
        "input_artifacts": input_artifacts,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "working_tree_hash": working_tree_hash,
        "resolved_config": dict(resolved_config or {}),
        "random_seed": random_seed,
        "environment_lock_hash": lock_hash,
    }
    digest = hashlib.sha256(
        json.dumps(reproducibility_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    run_id = f"{timestamp}_{safe_label}_{digest}"
    run_dir = _mkdir_unique(Path(root), run_id)
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "timestamp_utc": timestamp,
        "label": label,
        "hash": digest,
        "content_hash": digest,
        "inputs": [str(item) for item in inputs],
        "input_artifacts": input_artifacts,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "working_tree_hash": working_tree_hash,
        "resolved_config": dict(resolved_config or {}),
        "python_version": sys.version,
        "environment_lock_hash": lock_hash,
        "random_seed": random_seed,
        "command": list(command or sys.argv),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return HashedRunOutput(run_id=run_dir.name, run_dir=run_dir, manifest_path=manifest_path)


def path_in_run(path: str | Path, run_output: HashedRunOutput | None) -> Path:
    output_path = Path(path)
    if run_output is None or output_path.is_absolute():
        return output_path
    if output_path.parts and output_path.parts[0] == "outputs":
        output_path = Path(*output_path.parts[1:]) if len(output_path.parts) > 1 else Path(".")
    return run_output.run_dir / output_path


def _mkdir_unique(root: Path, run_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(100):
        candidate = root / run_id if attempt == 0 else root / f"{run_id}_{attempt:02d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not create a unique run directory under {root}")


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.strip()).strip("-._")
    return (slug or "run")[:48]


def _input_artifact(value: str | Path) -> dict[str, Any]:
    path = Path(value)
    if path.is_file():
        return {
            "kind": "file",
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    if path.is_dir():
        digest = hashlib.sha256()
        file_count = 0
        total_size = 0
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix()
            child_hash = _file_sha256(child)
            size = child.stat().st_size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(child_hash.encode("ascii"))
            digest.update(b"\0")
            file_count += 1
            total_size += size
        return {
            "kind": "directory",
            "path": str(path),
            "file_count": file_count,
            "size": total_size,
            "sha256": digest.hexdigest(),
        }
    text = str(value)
    return {
        "kind": "literal",
        "value": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_lock_hash() -> str:
    """Hash the strongest available environment definition.

    ``uv.lock`` is preferred when present. Source checkouts without the lock fall
    back to ``pyproject.toml`` instead of emitting a null reproducibility field.
    """

    for candidate in _ENVIRONMENT_LOCK_CANDIDATES:
        if candidate.is_file():
            return _file_sha256(candidate)
    return hashlib.sha256(b"unlocked-environment").hexdigest()


def _git_metadata() -> tuple[str | None, bool | None, str | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dirty = bool(status.strip())
        working_tree_hash = None
        if dirty:
            digest = hashlib.sha256()
            digest.update(
                subprocess.run(
                    ["git", "diff", "--binary", "HEAD"],
                    check=True,
                    capture_output=True,
                ).stdout
            )
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            for item in sorted(untracked):
                path = Path(item)
                if path.is_file():
                    digest.update(item.encode("utf-8"))
                    digest.update(b"\0")
                    digest.update(_file_sha256(path).encode("ascii"))
            working_tree_hash = digest.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return None, None, None
    return commit or None, dirty, working_tree_hash
