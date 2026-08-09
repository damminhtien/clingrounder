"""Registry for small, offline-safe artifacts shipped in the Python package.

Large terminology and model releases remain external artifacts.  This registry deliberately
contains only the small pack required by the public quickstart and never performs an implicit
network request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from clingrounder.artifacts.cache import ArtifactCache
from clingrounder.artifacts.manifest import (
    ArtifactManifest,
    ArtifactManifestError,
    fingerprint_payload,
    payload_size_bytes,
)

__all__ = [
    "ArtifactNotFoundError",
    "ArtifactManifest",
    "ArtifactManifestError",
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

        return fingerprint_payload(self.root)

    @property
    def profile_path(self) -> Path:
        return self.root / "profile.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def manifest(self) -> dict[str, object]:
        """Return metadata suitable for an experiment or prediction manifest."""

        return ArtifactManifest.read(self.manifest_path).as_dict()

    @property
    def payload_size_bytes(self) -> int:
        """Return the size covered by the pinned payload fingerprint."""

        return payload_size_bytes(self.root)

    def verify_manifest(self) -> None:
        """Fail closed when a bundled or cached pack differs from its declared manifest."""

        try:
            manifest = ArtifactManifest.read(self.manifest_path)
            manifest.validate_payload(self.root)
        except (OSError, ArtifactManifestError) as error:
            raise ArtifactNotFoundError(
                f"Invalid artifact manifest (checksum/size or contents): {self.manifest_path}"
            ) from error
        if manifest.artifact_id != self.artifact_id or manifest.revision != self.revision:
            raise ArtifactNotFoundError(
                f"Artifact manifest identity mismatch: {self.manifest_path}"
            )

    def install(self, cache_dir: str | Path) -> Path:
        """Copy the pack atomically into a caller-owned cache and return its root.

        The operation is intentionally explicit.  ``from_pretrained`` can use the bundled pack
        directly, while callers that need a stable external path may call ``Pipeline.download``.
        """

        self.verify_manifest()
        manifest = ArtifactManifest.read(self.manifest_path)
        return ArtifactCache(cache_dir).install(self.root, manifest)


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
