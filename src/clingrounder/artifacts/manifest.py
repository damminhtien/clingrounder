"""Typed manifests and deterministic fingerprints for reusable artifacts.

The manifest is deliberately independent of any registry or pipeline.  This keeps model,
terminology, and pipeline packs verifiable by the same small contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, TypeAlias

__all__ = [
    "ArtifactManifest",
    "ArtifactManifestError",
    "fingerprint_payload",
    "payload_size_bytes",
]


_SCHEMA_VERSION = "clingrounder.artifact-manifest.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ArtifactMetricValue: TypeAlias = str | float


class ArtifactManifestError(ValueError):
    """Raised when an artifact manifest or payload violates its contract."""


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Immutable identity, evidence, and payload metadata for one artifact release."""

    artifact_id: str
    revision: str
    artifact_type: str
    license: str
    sha256: str
    size_bytes: int
    contents: tuple[str, ...]
    metrics: tuple[tuple[str, ArtifactMetricValue], ...] = ()
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ArtifactManifestError(
                f"Unsupported artifact manifest schema: {self.schema_version!r}"
            )
        for field_name in ("artifact_id", "revision", "artifact_type", "license"):
            if not getattr(self, field_name).strip():
                raise ArtifactManifestError(f"{field_name} must be non-empty")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ArtifactManifestError("sha256 must be 64 lowercase hexadecimal characters")
        if self.size_bytes < 0:
            raise ArtifactManifestError("size_bytes must be non-negative")
        if not self.contents:
            raise ArtifactManifestError("contents must not be empty")
        if tuple(sorted(set(self.contents))) != self.contents:
            raise ArtifactManifestError("contents must be sorted and unique")
        for name in self.contents:
            _validate_relative_file_name(name)
        metric_names = tuple(name for name, _ in self.metrics)
        if tuple(sorted(set(metric_names))) != metric_names:
            raise ArtifactManifestError("metrics must have sorted, unique names")
        for name, value in self.metrics:
            if not name.strip():
                raise ArtifactManifestError("metric names must be non-empty")
            if isinstance(value, str):
                if not value.strip():
                    raise ArtifactManifestError(f"metric {name!r} must not be empty")
            elif (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ArtifactManifestError(f"metric {name!r} must be a finite number or string")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ArtifactManifest":
        """Parse the public JSON shape without silently accepting unknown structure."""

        try:
            schema_version = _required_string(payload, "schema_version")
            artifact = payload["artifact"]
            contents = payload["contents"]
            if not isinstance(artifact, Mapping) or not isinstance(contents, list):
                raise TypeError
            _expect_keys(
                payload,
                {"schema_version", "artifact", "contents"},
                "manifest",
                optional={"metrics"},
            )
            _expect_keys(
                artifact,
                {"id", "version", "type", "license", "sha256", "size_bytes"},
                "artifact",
            )
            raw_metrics = payload.get("metrics", {})
            if not isinstance(raw_metrics, Mapping):
                raise ArtifactManifestError("metrics must be an object")
            metrics: list[tuple[str, ArtifactMetricValue]] = []
            for name, value in raw_metrics.items():
                if not isinstance(name, str) or not name.strip():
                    raise ArtifactManifestError("metric names must be non-empty strings")
                if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                    raise ArtifactManifestError(
                        f"metric {name!r} must be a finite number or string"
                    )
                metrics.append(
                    (name, float(value) if isinstance(value, (int, float)) else value)
                )
            values = {
                "artifact_id": _required_string(artifact, "id"),
                "revision": _required_string(artifact, "version"),
                "artifact_type": _required_string(artifact, "type"),
                "license": _required_string(artifact, "license"),
                "sha256": _required_string(artifact, "sha256"),
                "size_bytes": artifact["size_bytes"],
                "contents": tuple(contents),
                "metrics": tuple(sorted(metrics)),
                "schema_version": schema_version,
            }
        except ArtifactManifestError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactManifestError("Malformed artifact manifest") from error
        if not isinstance(values["size_bytes"], int) or isinstance(values["size_bytes"], bool):
            raise ArtifactManifestError("artifact.size_bytes must be an integer")
        if any(not isinstance(item, str) for item in values["contents"]):
            raise ArtifactManifestError("contents must contain only file names")
        return cls(
            artifact_id=values["artifact_id"],
            revision=values["revision"],
            artifact_type=values["artifact_type"],
            license=values["license"],
            sha256=values["sha256"],
            size_bytes=values["size_bytes"],
            contents=values["contents"],
            metrics=values["metrics"],
            schema_version=values["schema_version"],
        )

    @classmethod
    def read(cls, path: str | Path) -> "ArtifactManifest":
        """Read one manifest from disk using UTF-8 JSON."""

        manifest_path = Path(path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ArtifactManifestError(f"Unable to read artifact manifest: {manifest_path}") from error
        if not isinstance(payload, Mapping):
            raise ArtifactManifestError("Artifact manifest root must be an object")
        return cls.from_mapping(payload)

    def as_dict(self) -> dict[str, object]:
        """Return the stable on-disk representation."""

        return {
            "schema_version": self.schema_version,
            "artifact": {
                "id": self.artifact_id,
                "version": self.revision,
                "type": self.artifact_type,
                "license": self.license,
                "sha256": self.sha256,
                "size_bytes": self.size_bytes,
            },
            "contents": list(self.contents),
            "metrics": {name: value for name, value in self.metrics},
        }

    def validate_payload(self, root: str | Path) -> None:
        """Verify payload files, size, and digest against this manifest.

        INVARIANT: the manifest excludes ``manifest.json`` from the payload digest so the
        manifest can record its own payload fingerprint without a circular hash.
        """

        payload_root = Path(root)
        if not payload_root.is_dir():
            raise ArtifactManifestError(f"Artifact payload is not a directory: {payload_root}")
        actual_files = _payload_files(payload_root)
        expected_files = tuple(self.contents)
        if actual_files != expected_files:
            missing = sorted(set(expected_files) - set(actual_files))
            unexpected = sorted(set(actual_files) - set(expected_files))
            raise ArtifactManifestError(
                f"Artifact contents mismatch (missing={missing}, unexpected={unexpected})"
            )
        actual_size = payload_size_bytes(payload_root)
        actual_sha = fingerprint_payload(payload_root)
        if actual_size != self.size_bytes:
            raise ArtifactManifestError(
                f"Artifact size mismatch: expected {self.size_bytes}, got {actual_size}"
            )
        if actual_sha != self.sha256:
            raise ArtifactManifestError(
                f"Artifact checksum mismatch: expected {self.sha256}, got {actual_sha}"
            )


def fingerprint_payload(root: str | Path) -> str:
    """Return a deterministic SHA-256 over relative names and file bytes."""

    digest = hashlib.sha256()
    payload_root = Path(root)
    for relative_name in _payload_files(payload_root):
        path = payload_root / relative_name
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def payload_size_bytes(root: str | Path) -> int:
    """Return the total byte size of all regular payload files."""

    return sum((Path(root) / name).stat().st_size for name in _payload_files(Path(root)))


def _payload_files(root: Path) -> tuple[str, ...]:
    if not root.is_dir():
        return ()
    files: list[str] = []
    for path in root.rglob("*"):
        if path.name == "manifest.json":
            continue
        if path.is_symlink():
            raise ArtifactManifestError(f"Symlinks are not allowed in artifacts: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            _validate_relative_file_name(relative)
            files.append(relative)
    return tuple(sorted(files))


def _validate_relative_file_name(name: str) -> None:
    path = Path(name)
    if not name or path.is_absolute() or "\\" in name or ".." in path.parts:
        raise ArtifactManifestError(f"Invalid artifact file name: {name!r}")
    if name == "manifest.json":
        raise ArtifactManifestError("manifest.json is reserved for artifact metadata")


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise ArtifactManifestError(f"{key} must be a non-empty string")
    return value


def _expect_keys(
    mapping: Mapping[str, object],
    expected: set[str],
    label: str,
    *,
    optional: set[str] | None = None,
) -> None:
    allowed = expected | (optional or set())
    unknown = sorted(set(mapping) - allowed)
    missing = sorted(expected - set(mapping))
    if unknown or missing:
        raise ArtifactManifestError(
            f"Malformed {label} manifest fields (unknown={unknown}, missing={missing})"
        )
