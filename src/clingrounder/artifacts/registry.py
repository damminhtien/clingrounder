"""Registry for small, offline-safe artifacts shipped in the Python package.

Large terminology and model releases remain external artifacts.  This registry deliberately
contains only the small pack required by the public quickstart and never performs an implicit
network request.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Final

__all__ = [
    "ArtifactNotFoundError",
    "BuiltinArtifact",
    "get_builtin_artifact",
    "list_builtin_artifacts",
]


class ArtifactNotFoundError(LookupError):
    """Raised when a built-in artifact name or revision is not available."""


@dataclass(frozen=True, slots=True)
class BuiltinArtifact:
    """Metadata and safe operations for one package-bundled resource pack."""

    artifact_id: str
    revision: str
    root: Path
    license: str = "MIT"

    @property
    def fingerprint(self) -> str:
        """Return a deterministic digest over all pack files and their relative names."""

        return _fingerprint_root(self.root)

    @property
    def profile_path(self) -> Path:
        return self.root / "profile.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def manifest(self) -> dict[str, object]:
        """Return metadata suitable for an experiment or prediction manifest."""

        return {
            "schema_version": "clingrounder.artifact-manifest.v1",
            "artifact": {
                "id": self.artifact_id,
                "version": self.revision,
                "type": "pipeline-pack",
                "license": self.license,
                "sha256": self.fingerprint,
                "size_bytes": self.payload_size_bytes,
            },
            "contents": sorted(path.name for path in self.root.iterdir() if path.is_file()),
        }

    @property
    def payload_size_bytes(self) -> int:
        """Return the size covered by the pinned payload fingerprint."""

        return sum(
            path.stat().st_size
            for path in self.root.iterdir()
            if path.is_file() and path.name != "manifest.json"
        )

    def verify_manifest(self) -> None:
        """Fail closed when a bundled or cached pack differs from its declared manifest."""

        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            artifact = payload["artifact"]
            expected_id = artifact["id"]
            expected_revision = artifact["version"]
            expected_sha = artifact["sha256"]
            expected_size = artifact["size_bytes"]
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ArtifactNotFoundError(
                f"Invalid artifact manifest: {self.manifest_path}"
            ) from error
        if expected_id != self.artifact_id or expected_revision != self.revision:
            raise ArtifactNotFoundError(
                f"Artifact manifest identity mismatch: {self.manifest_path}"
            )
        if expected_sha != self.fingerprint or expected_size != self.payload_size_bytes:
            raise ArtifactNotFoundError(
                f"Artifact checksum/size mismatch: {self.manifest_path}"
            )

    def install(self, cache_dir: str | Path) -> Path:
        """Copy the pack atomically into a caller-owned cache and return its root.

        The operation is intentionally explicit.  ``from_pretrained`` can use the bundled pack
        directly, while callers that need a stable external path may call ``Pipeline.download``.
        """

        destination = Path(cache_dir).expanduser() / self.artifact_id / self.revision
        temporary = destination.with_name(f".{destination.name}.tmp")
        if destination.exists():
            if _fingerprint_root(destination) == self.fingerprint:
                return destination
            shutil.rmtree(destination)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(self.root, temporary)
        temporary.replace(destination)
        return destination


_PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent
_ARTIFACTS: Final[dict[str, BuiltinArtifact]] = {
    "vi-clinical-small": BuiltinArtifact(
        artifact_id="vi-clinical-small",
        revision="2026.08",
        root=_PACKAGE_ROOT / "packs" / "vi-clinical-small",
    )
}


def list_builtin_artifacts() -> tuple[BuiltinArtifact, ...]:
    """List bundled artifacts in deterministic ID order."""

    return tuple(_ARTIFACTS[name] for name in sorted(_ARTIFACTS))


def get_builtin_artifact(name: str, revision: str | None = None) -> BuiltinArtifact:
    """Resolve one bundled artifact without network fallback."""

    try:
        artifact = _ARTIFACTS[name]
    except KeyError as error:
        available = ", ".join(sorted(_ARTIFACTS))
        raise ArtifactNotFoundError(
            f"Unknown artifact {name!r}; available bundled artifacts: {available}"
        ) from error
    if revision is not None and revision != artifact.revision:
        raise ArtifactNotFoundError(
            f"Artifact {name!r} has revision {artifact.revision!r}, not {revision!r}"
        )
    if not artifact.root.is_dir():
        raise ArtifactNotFoundError(f"Bundled artifact files are missing: {artifact.root}")
    artifact.verify_manifest()
    return artifact


def _fingerprint_root(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.iterdir() if item.is_file() and item.name != "manifest.json"
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
