"""Explicit local-file connector for licensed archives supplied by the user."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote, unquote, urlparse

from medical_kg_nlp.mining.connectors.base import RegisteredConnectorAdapter
from medical_kg_nlp.mining.records import DiscoveredArtifact, SourceRequest
from medical_kg_nlp.mining.registry import SourceDefinition

__all__ = ["LocalArchiveConnector", "LocalFileTransport"]


class LocalFileTransport:
    """Open only local paths and ``file:`` URIs."""

    def open(self, uri: str) -> BinaryIO:
        parsed = urlparse(uri)
        if parsed.scheme not in {"", "file"}:
            raise ValueError(f"Local transport does not support URI scheme {parsed.scheme!r}")
        path = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(uri)
        return path.open("rb")


class LocalArchiveConnector(RegisteredConnectorAdapter):
    """Discover an explicit list of local files without scanning arbitrary directories."""

    connector_revision = "2"

    def __init__(self, source: SourceDefinition) -> None:
        super().__init__(source, LocalFileTransport())

    def _discover(self, request: SourceRequest) -> Iterable[DiscoveredArtifact]:
        paths = _string_sequence(request.parameters.get("paths"), field_name="paths")
        media_type = str(request.parameters.get("media_type", "application/octet-stream"))
        checksums = _string_mapping(request.parameters.get("sha256", {}), field_name="sha256")
        for raw_path in sorted(paths):
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_sha256 = checksums.get(raw_path) or checksums.get(path.name)
            if expected_sha256 is None:
                expected_sha256 = _file_sha256(path)
            yield DiscoveredArtifact(
                source_id=request.source_id,
                source_version=request.source_version,
                uri=path.as_uri(),
                media_type=media_type,
                expected_sha256=expected_sha256,
                metadata={"filename": path.name, "byte_size": str(path.stat().st_size)},
            )

    def _persisted_source_uri(self, artifact: DiscoveredArtifact) -> str:
        """Replace a workstation path with a source-scoped portable locator."""

        filename = artifact.metadata.get("filename")
        if not filename:
            raise ValueError("Local artifacts require filename metadata")
        # PRIVACY: the actual file URI is needed only by LocalFileTransport. Persisting it
        # would bind manifests to one home directory and can disclose controlled mounts.
        return f"local-source://{quote(artifact.source_id, safe='')}/{quote(filename, safe='')}"


def _string_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return result


def _string_mapping(value: Any, *, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return {str(key): str(item) for key, item in value.items()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
