"""Explicit local artifact acquisition.

The reusable core intentionally has no implicit HTTP client.  A future remote provider can
implement the same manifest/cache contract without making network access part of ``load_pipeline``.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from clingrounder.artifacts.cache import ArtifactCache
from clingrounder.artifacts.manifest import ArtifactManifest

__all__ = ["ArtifactDownloadError", "ArtifactDownloader"]


class ArtifactDownloadError(RuntimeError):
    """Raised when an explicit artifact source cannot be used safely."""


class ArtifactDownloader:
    """Download/materialize artifacts from local paths or ``file://`` URIs only."""

    def materialize(
        self,
        source: str | Path,
        manifest: ArtifactManifest,
        cache: ArtifactCache,
    ) -> Path:
        """Verify a local source and install it atomically into ``cache``."""

        source_path = _local_source(source)
        if not source_path.is_dir():
            raise ArtifactDownloadError(f"Local artifact source is not a directory: {source_path}")
        try:
            return cache.install(source_path, manifest)
        except RuntimeError as error:
            raise ArtifactDownloadError(str(error)) from error


def _local_source(source: str | Path) -> Path:
    raw = str(source)
    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        raise ArtifactDownloadError(
            f"Unsupported artifact source scheme {parsed.scheme!r}; core accepts local/file sources only"
        )
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ArtifactDownloadError("Remote file hosts are not allowed")
        return Path(unquote(parsed.path)).expanduser().resolve()
    return Path(source).expanduser().resolve()
