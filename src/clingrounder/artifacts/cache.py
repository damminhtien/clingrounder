"""Versioned, caller-owned cache for verified artifact payloads."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile

from clingrounder.artifacts.manifest import ArtifactManifest

__all__ = ["ArtifactCache", "ArtifactCacheError"]


class ArtifactCacheError(RuntimeError):
    """Raised when a cache operation would be unsafe or inconsistent."""


@dataclass(frozen=True, slots=True)
class ArtifactCache:
    """Materialize verified artifacts under ``<root>/<id>/<revision>``."""

    root: Path

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "root", Path(root).expanduser().resolve())

    def path_for(self, manifest: ArtifactManifest) -> Path:
        """Return the deterministic cache path for one artifact identity."""

        _validate_component(manifest.artifact_id, "artifact_id")
        _validate_component(manifest.revision, "revision")
        return self.root / manifest.artifact_id / manifest.revision

    def resolve(self, manifest: ArtifactManifest) -> Path | None:
        """Return a verified cache hit, or ``None`` when it is absent."""

        destination = self.path_for(manifest)
        if not destination.exists():
            return None
        try:
            manifest.validate_payload(destination)
        except (OSError, ValueError) as error:
            raise ArtifactCacheError(f"Cached artifact failed verification: {destination}") from error
        return destination

    def install(self, source: str | Path, manifest: ArtifactManifest) -> Path:
        """Verify and atomically copy a local artifact into this cache.

        SCALING: cache identities include revision, so installing a new release never invalidates
        or overwrites a previously verified release.  A temporary sibling is removed on failure.
        """

        source_root = Path(source).expanduser().resolve()
        try:
            manifest.validate_payload(source_root)
        except (OSError, ValueError) as error:
            raise ArtifactCacheError(f"Source artifact failed verification: {source_root}") from error
        existing = self.resolve(manifest)
        if existing is not None:
            return existing

        destination = self.path_for(manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{manifest.revision}.", dir=destination.parent))
        try:
            for relative_name in manifest.contents:
                target = temporary / relative_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / relative_name, target)
            # Keep the manifest beside the cached payload for offline inspection and later
            # verification.  It is metadata, so it is intentionally excluded from the digest.
            source_manifest = source_root / "manifest.json"
            if source_manifest.is_file():
                shutil.copy2(source_manifest, temporary / "manifest.json")
            manifest.validate_payload(temporary)
            # INVARIANT: rename is the publication point; readers see either no cache entry or
            # a complete verified directory, never a partially copied payload.
            os.rename(temporary, destination)
            return destination
        except FileExistsError:
            cached = self.resolve(manifest)
            if cached is not None:
                return cached
            raise ArtifactCacheError(f"Concurrent cache install produced an invalid entry: {destination}")
        except (OSError, ValueError) as error:
            raise ArtifactCacheError(f"Unable to install artifact at {destination}") from error
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)


def _validate_component(value: str, field_name: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ArtifactCacheError(f"Unsafe {field_name}: {value!r}")
