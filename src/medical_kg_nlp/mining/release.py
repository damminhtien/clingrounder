"""Portable, content-addressed locks for materialized mining releases.

Mining outputs deliberately live outside Git because corpora and derived indexes can be
large or license-restricted.  A dataset snapshot alone is not enough to reproduce a
NER/retrieval experiment on another machine: the model dataset, terminology overlays,
benchmarks, source policies, and code dependency lock must agree.  This module records
that contract without embedding host-specific paths.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text

__all__ = [
    "LoadedMiningReleaseSpec",
    "MiningReleaseLock",
    "MiningReleaseSpec",
    "ReleaseArtifactSpec",
    "ReleaseRebuildStep",
    "build_mining_release_lock",
    "load_mining_release_spec",
    "verify_mining_release_lock",
]

_LOCK_SCHEMA_VERSION = "medical-mining-release-lock.v1"


class ReleaseArtifactSpec(BaseModel):
    """One repository-relative artifact required by a reproducible release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    path: str = Field(min_length=1)
    description: str = Field(min_length=1)
    required: bool = True
    exclude: tuple[str, ...] = ()

    @field_validator("path")
    @classmethod
    def validate_portable_path(cls, value: str) -> str:
        """Reject paths that would make a lock specific to one workstation."""

        return _portable_artifact_path(value)

    @field_validator("exclude")
    @classmethod
    def validate_excludes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_portable_glob(value) for value in values)


class ReleaseRebuildStep(BaseModel):
    """A documented command in the release reconstruction sequence.

    Steps are intentionally not executed by the lock command.  Acquisition may require
    explicit licence acceptance, external storage, or a GPU, so reconstruction remains
    reviewable and controlled by the caller.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    command: str = Field(min_length=1)

    @field_validator("command")
    @classmethod
    def validate_portable_command(cls, value: str) -> str:
        command = value.strip()
        if "\n" in command or "\r" in command or "\x00" in command:
            raise ValueError("rebuild commands must be single-line text")
        # PRIVACY: a checked-in release specification must never leak a user's home path.
        if "/Users/" in command or "\\Users\\" in command or "/home/" in command:
            raise ValueError("rebuild commands must not contain a workstation-local path")
        return command


class MiningReleaseSpec(BaseModel):
    """Checked-in declaration for a portable NER/retrieval data release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["medical-mining-release-spec.v1"]
    release_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    release_root: str = Field(min_length=1)
    artifacts: tuple[ReleaseArtifactSpec, ...] = Field(min_length=1)
    rebuild_steps: tuple[ReleaseRebuildStep, ...] = ()

    @field_validator("release_root")
    @classmethod
    def validate_relative_root(cls, value: str) -> str:
        root = value.strip()
        if not root:
            raise ValueError("release_root must be non-empty")
        if Path(root).is_absolute() or PureWindowsPath(root).is_absolute():
            raise ValueError("release_root must be relative to the release specification")
        return root

    @model_validator(mode="after")
    def validate_unique_identifiers(self) -> "MiningReleaseSpec":
        artifact_ids = [artifact.id for artifact in self.artifacts]
        step_ids = [step.id for step in self.rebuild_steps]
        all_ids = [*artifact_ids, *step_ids]
        duplicates = sorted(
            {value for value in all_ids if all_ids.count(value) > 1}
        )
        if duplicates:
            raise ValueError(f"release artifact and step IDs must be unique: {', '.join(duplicates)}")
        excluded_non_code = [
            artifact.id
            for artifact in self.artifacts
            if artifact.exclude and artifact.role != "implementation"
        ]
        if excluded_non_code:
            # INVARIANT: no release may hide corpus, terminology, benchmark, or model bytes.
            raise ValueError(
                "exclude patterns are allowed only for implementation artifacts: "
                + ", ".join(excluded_non_code)
            )
        return self


class LockedReleaseArtifact(BaseModel):
    """Content fingerprint captured for one artifact at release-lock time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    path: str = Field(min_length=1)
    description: str = Field(min_length=1)
    required: bool
    exclude: tuple[str, ...] = ()
    present: bool
    kind: Literal["file", "directory", "absent"]
    sha256: str | None = None
    byte_size: int = Field(ge=0)
    file_count: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_portable_path(cls, value: str) -> str:
        return _portable_artifact_path(value)

    @field_validator("exclude")
    @classmethod
    def validate_excludes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_portable_glob(value) for value in values)

    @model_validator(mode="after")
    def validate_presence_metadata(self) -> "LockedReleaseArtifact":
        if self.present:
            if self.kind == "absent" or not _is_sha256(self.sha256):
                raise ValueError("present artifacts require a non-absent kind and SHA-256")
        elif self.kind != "absent" or self.sha256 is not None or self.file_count != 0:
            raise ValueError("absent artifacts cannot carry content metadata")
        return self


class MiningReleaseLock(BaseModel):
    """Portable lock manifest whose paths are relative to a caller-provided root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["medical-mining-release-lock.v1"]
    release_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    spec: dict[str, str]
    artifacts: tuple[LockedReleaseArtifact, ...] = Field(min_length=1)
    rebuild_steps: tuple[ReleaseRebuildStep, ...] = ()
    release_fingerprint: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_lock(self) -> "MiningReleaseLock":
        spec_path = self.spec.get("path")
        spec_sha256 = self.spec.get("sha256")
        if not isinstance(spec_path, str):
            raise ValueError("release lock spec.path must be a string")
        _portable_artifact_path(spec_path)
        if not _is_sha256(spec_sha256):
            raise ValueError("release lock spec.sha256 must be a lowercase SHA-256")
        if not _is_sha256(self.release_fingerprint):
            raise ValueError("release_fingerprint must be a lowercase SHA-256")
        artifact_ids = [artifact.id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("release lock contains duplicate artifact IDs")
        return self


@dataclass(frozen=True)
class LoadedMiningReleaseSpec:
    """A parsed release spec with its repository root resolved exactly once."""

    spec: MiningReleaseSpec
    spec_path: Path
    release_root: Path


@dataclass(frozen=True)
class _PathFingerprint:
    """Internal deterministic fingerprint for one file or directory."""

    kind: Literal["file", "directory"]
    sha256: str
    byte_size: int
    file_count: int


def load_mining_release_spec(path: str | Path) -> LoadedMiningReleaseSpec:
    """Load a strict spec and resolve its declared root without reading artifacts."""

    spec_path = Path(path).resolve()
    with spec_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{spec_path}: release specification must be a mapping")
    spec = MiningReleaseSpec.model_validate(raw)
    release_root = (spec_path.parent / spec.release_root).resolve()
    if not release_root.is_dir():
        raise ValueError(f"{spec_path}: release_root does not exist: {spec.release_root}")
    _relative_to_root(spec_path, release_root)
    return LoadedMiningReleaseSpec(
        spec=spec,
        spec_path=spec_path,
        release_root=release_root,
    )


def build_mining_release_lock(
    spec_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write a deterministic lock for all declared materialized artifacts.

    The output belongs outside every declared artifact path.  Otherwise a pre-existing
    lock could hash itself through a directory fingerprint and become non-deterministic.
    """

    loaded = load_mining_release_spec(spec_path)
    output = Path(output_path).resolve()
    output_relative = _relative_to_root(output, loaded.release_root)
    for artifact in loaded.spec.artifacts:
        if _paths_intersect(output_relative, artifact.path):
            raise ValueError(
                "release lock output must not be inside a declared artifact path: "
                f"{artifact.id}"
            )

    artifacts = tuple(
        _lock_artifact(loaded.release_root, artifact)
        for artifact in loaded.spec.artifacts
    )
    payload: dict[str, Any] = {
        "schema_version": _LOCK_SCHEMA_VERSION,
        "release_id": loaded.spec.release_id,
        "description": loaded.spec.description,
        "spec": {
            "path": _relative_to_root(loaded.spec_path, loaded.release_root),
            "sha256": sha256_file(loaded.spec_path),
        },
        "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
        "rebuild_steps": [step.model_dump(mode="json") for step in loaded.spec.rebuild_steps],
    }
    payload["release_fingerprint"] = _release_fingerprint(payload)
    MiningReleaseLock.model_validate(payload)
    write_json(output, payload)
    return payload


def verify_mining_release_lock(
    manifest_path: str | Path,
    *,
    release_root: str | Path,
    require_optional: bool = False,
) -> dict[str, Any]:
    """Verify a release lock after checkout, artifact restore, or data reconstruction."""

    manifest = _load_lock(manifest_path)
    root = Path(release_root).resolve()
    if not root.is_dir():
        raise ValueError("release_root must be an existing directory")

    errors: list[str] = []
    optional_missing: list[str] = []
    verified_artifacts = 0
    payload_without_fingerprint = manifest.model_dump(mode="json")
    expected_fingerprint = payload_without_fingerprint.pop("release_fingerprint")
    if _release_fingerprint(payload_without_fingerprint) != expected_fingerprint:
        errors.append("invalid_release_fingerprint")

    _verify_spec(manifest, root, errors)
    for artifact in manifest.artifacts:
        target = _resolve_artifact_path(root, artifact.path)
        if not target.exists():
            if artifact.required or require_optional:
                errors.append(f"missing_artifact:{artifact.id}")
            else:
                optional_missing.append(artifact.id)
            continue
        if not artifact.present:
            errors.append(f"unexpected_optional_artifact:{artifact.id}")
            continue
        try:
            observed = _fingerprint_path(target, exclude=artifact.exclude)
        except ValueError as error:
            errors.append(f"invalid_artifact:{artifact.id}:{error}")
            continue
        if observed.kind != artifact.kind:
            errors.append(f"kind_mismatch:{artifact.id}")
        if observed.sha256 != artifact.sha256:
            errors.append(f"sha256_mismatch:{artifact.id}")
        if observed.byte_size != artifact.byte_size:
            errors.append(f"byte_size_mismatch:{artifact.id}")
        if observed.file_count != artifact.file_count:
            errors.append(f"file_count_mismatch:{artifact.id}")
        if (
            observed.kind == artifact.kind
            and observed.sha256 == artifact.sha256
            and observed.byte_size == artifact.byte_size
            and observed.file_count == artifact.file_count
        ):
            verified_artifacts += 1

    return {
        "schema_version": "medical-mining-release-verification.v1",
        "release_id": manifest.release_id,
        "release_fingerprint": manifest.release_fingerprint,
        "artifact_count": len(manifest.artifacts),
        "verified_artifact_count": verified_artifacts,
        "optional_missing_artifact_ids": sorted(optional_missing),
        "errors": sorted(set(errors)),
        "valid": not errors,
    }


def _lock_artifact(root: Path, artifact: ReleaseArtifactSpec) -> LockedReleaseArtifact:
    target = _resolve_artifact_path(root, artifact.path)
    if not target.exists():
        if artifact.required:
            raise FileNotFoundError(f"required release artifact is missing: {artifact.id}")
        return LockedReleaseArtifact(
            id=artifact.id,
            role=artifact.role,
            path=artifact.path,
            description=artifact.description,
            required=False,
            exclude=artifact.exclude,
            present=False,
            kind="absent",
            sha256=None,
            byte_size=0,
            file_count=0,
        )
    fingerprint = _fingerprint_path(target, exclude=artifact.exclude)
    return LockedReleaseArtifact(
        id=artifact.id,
        role=artifact.role,
        path=artifact.path,
        description=artifact.description,
        required=artifact.required,
        exclude=artifact.exclude,
        present=True,
        kind=fingerprint.kind,
        sha256=fingerprint.sha256,
        byte_size=fingerprint.byte_size,
        file_count=fingerprint.file_count,
    )


def _verify_spec(manifest: MiningReleaseLock, root: Path, errors: list[str]) -> None:
    spec_path = _resolve_artifact_path(root, manifest.spec["path"])
    if not spec_path.is_file():
        errors.append("missing_release_spec")
        return
    if sha256_file(spec_path) != manifest.spec["sha256"]:
        errors.append("release_spec_sha256_mismatch")


def _load_lock(path: str | Path) -> MiningReleaseLock:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: release lock must be a mapping")
    return MiningReleaseLock.model_validate(raw)


def _fingerprint_path(
    path: Path,
    *,
    exclude: tuple[str, ...] = (),
) -> _PathFingerprint:
    if path.is_symlink():
        raise ValueError("symlink artifacts are not portable")
    if path.is_file():
        if exclude:
            raise ValueError("exclude patterns are valid only for directory artifacts")
        return _PathFingerprint(
            kind="file",
            sha256=sha256_file(path),
            byte_size=path.stat().st_size,
            file_count=1,
        )
    if not path.is_dir():
        raise ValueError("artifact is neither a regular file nor a directory")

    digest = hashlib.sha256()
    digest.update(b"medical-mining-directory-v1\0")
    byte_size = 0
    file_count = 0
    # SCALING: walk and hash one member at a time so a multi-GB artifact directory never
    # needs a full in-memory inventory.  Directory names are included for structural safety.
    for current_root, directory_names, file_names in os.walk(path, topdown=True, followlinks=False):
        current = Path(current_root)
        directory_names.sort()
        file_names.sort()
        retained_directories: list[str] = []
        for directory_name in directory_names:
            child = current / directory_name
            if child.is_symlink():
                raise ValueError("directory contains a symlink")
            relative = child.relative_to(path).as_posix()
            if _matches_exclude(relative, exclude):
                continue
            retained_directories.append(directory_name)
            digest.update(f"D\0{relative}\0".encode("utf-8"))
        # SCALING: pruning ignored build caches prevents walking machine-specific trees.
        directory_names[:] = retained_directories
        for file_name in file_names:
            child = current / file_name
            if child.is_symlink() or not child.is_file():
                raise ValueError("directory contains a non-regular file")
            relative = child.relative_to(path).as_posix()
            if _matches_exclude(relative, exclude):
                continue
            file_sha256 = sha256_file(child)
            size = child.stat().st_size
            digest.update(f"F\0{relative}\0{size}\0{file_sha256}\0".encode("utf-8"))
            byte_size += size
            file_count += 1
    return _PathFingerprint(
        kind="directory",
        sha256=digest.hexdigest(),
        byte_size=byte_size,
        file_count=file_count,
    )


def _portable_artifact_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    if not path or path == ".":
        raise ValueError("artifact paths must name a file or directory below release_root")
    if Path(path).is_absolute() or PureWindowsPath(path).is_absolute():
        raise ValueError("artifact paths must be relative")
    parts = tuple(Path(path).parts)
    if ".." in parts or any(not part or part == "." for part in parts):
        raise ValueError("artifact paths must not escape release_root")
    return Path(*parts).as_posix()


def _portable_glob(value: str) -> str:
    pattern = value.strip().replace("\\", "/")
    if not pattern or pattern.startswith("/") or PureWindowsPath(pattern).is_absolute():
        raise ValueError("exclude patterns must be portable relative globs")
    if ".." in PurePosixPath(pattern).parts:
        raise ValueError("exclude patterns must not escape the artifact directory")
    return pattern


def _matches_exclude(relative_path: str, patterns: tuple[str, ...]) -> bool:
    candidate = PurePosixPath(relative_path)
    return any(candidate.match(pattern) for pattern in patterns)


def _resolve_artifact_path(root: Path, portable_path: str) -> Path:
    target = (root / portable_path).resolve()
    _relative_to_root(target, root)
    return target


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("release path escapes declared release_root") from error


def _paths_intersect(left: str, right: str) -> bool:
    left_parts = tuple(Path(left).parts)
    right_parts = tuple(Path(right).parts)
    return left_parts[: len(right_parts)] == right_parts or right_parts[: len(left_parts)] == left_parts


def _release_fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
