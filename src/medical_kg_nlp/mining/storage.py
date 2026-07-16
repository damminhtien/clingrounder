"""Content-addressed artifact stores for local disks and optional object storage."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO, cast

from medical_kg_nlp.mining.records import StoredObject

__all__ = ["FsspecArtifactStore", "LocalArtifactStore"]

_CHUNK_SIZE = 1024 * 1024


class LocalArtifactStore:
    """Store immutable objects under ``objects/sha256`` using atomic replacement."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.object_root = self.root / "objects" / "sha256"
        self.metadata_root = self.root / "metadata" / "sha256"
        self.tmp_root = self.root / "tmp"
        self.object_root.mkdir(parents=True, exist_ok=True)
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def put_stream(self, stream: BinaryIO, *, metadata: Mapping[str, str]) -> StoredObject:
        digest = hashlib.sha256()
        byte_size = 0
        descriptor, temporary_name = tempfile.mkstemp(prefix="artifact-", dir=self.tmp_root)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                while chunk := stream.read(_CHUNK_SIZE):
                    digest.update(chunk)
                    output.write(chunk)
                    byte_size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            sha256 = digest.hexdigest()
            object_path = self._object_path(sha256)
            object_path.parent.mkdir(parents=True, exist_ok=True)
            # SCALING: concurrent writers may race, but identical hashes make either byte stream
            # authoritative. os.replace keeps readers from observing partial artifacts.
            if object_path.exists():
                temporary_path.unlink()
            else:
                os.replace(temporary_path, object_path)
            self._write_metadata(sha256, byte_size=byte_size, metadata=metadata)
            return StoredObject(
                sha256=sha256,
                uri=object_path.resolve().as_uri(),
                byte_size=byte_size,
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    def open(self, sha256: str) -> BinaryIO:
        return self._object_path(sha256).open("rb")

    def exists(self, sha256: str) -> bool:
        return self._object_path(sha256).is_file()

    def _object_path(self, sha256: str) -> Path:
        _validate_digest(sha256)
        return self.object_root / sha256[:2] / sha256

    def _write_metadata(
        self,
        sha256: str,
        *,
        byte_size: int,
        metadata: Mapping[str, str],
    ) -> None:
        path = self.metadata_root / sha256[:2] / f"{sha256}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sha256": sha256,
            "byte_size": byte_size,
            "metadata": dict(sorted(metadata.items())),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)


class FsspecArtifactStore:
    """Content-addressed store for S3-compatible or other fsspec filesystems.

    ``fsspec`` and the backend implementation (for example ``s3fs``) are imported lazily so the
    clinical NLP core remains dependency-light.
    """

    def __init__(self, root_uri: str, *, storage_options: Mapping[str, Any] | None = None) -> None:
        try:
            fsspec = importlib.import_module("fsspec")
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install medical-kg-nlp[data] to use object storage") from error
        filesystem, root = fsspec.core.url_to_fs(root_uri, **dict(storage_options or {}))
        self._filesystem = filesystem
        self._root = str(root).rstrip("/")
        self._protocol = str(filesystem.protocol[0] if isinstance(filesystem.protocol, tuple) else filesystem.protocol)

    def put_stream(self, stream: BinaryIO, *, metadata: Mapping[str, str]) -> StoredObject:
        digest = hashlib.sha256()
        byte_size = 0
        with tempfile.NamedTemporaryFile(prefix="medical-kg-artifact-", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
                temporary.write(chunk)
                byte_size += len(chunk)
        try:
            sha256 = digest.hexdigest()
            destination = self._object_path(sha256)
            if not self._filesystem.exists(destination):
                self._filesystem.put_file(str(temporary_path), destination)
            metadata_path = self._metadata_path(sha256)
            if not self._filesystem.exists(metadata_path):
                with self._filesystem.open(metadata_path, "wb") as handle:
                    payload = json.dumps(
                        {
                            "sha256": sha256,
                            "byte_size": byte_size,
                            "metadata": dict(sorted(metadata.items())),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                    handle.write(payload)
            return StoredObject(
                sha256=sha256,
                uri=f"{self._protocol}://{destination}",
                byte_size=byte_size,
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    def open(self, sha256: str) -> BinaryIO:
        return cast(BinaryIO, self._filesystem.open(self._object_path(sha256), "rb"))

    def exists(self, sha256: str) -> bool:
        return bool(self._filesystem.exists(self._object_path(sha256)))

    def _object_path(self, sha256: str) -> str:
        _validate_digest(sha256)
        return f"{self._root}/objects/sha256/{sha256[:2]}/{sha256}"

    def _metadata_path(self, sha256: str) -> str:
        return f"{self._root}/metadata/sha256/{sha256[:2]}/{sha256}.json"


def _validate_digest(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("Expected a lowercase SHA-256 digest")
