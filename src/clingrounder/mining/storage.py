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

from clingrounder.mining.ports import ArtifactStorePort
from clingrounder.mining.records import StoredObject, content_addressed_object_uri

__all__ = ["FsspecArtifactStore", "LocalArtifactStore", "materialize_stored_object"]

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
                # INVARIANT: artifact manifests identify bytes, not workstation mounts.
                uri=content_addressed_object_uri(sha256),
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
            raise RuntimeError("Install clingrounder[data] to use object storage") from error
        filesystem, root = fsspec.core.url_to_fs(root_uri, **dict(storage_options or {}))
        self._filesystem = filesystem
        self._root = str(root).rstrip("/")

    def put_stream(self, stream: BinaryIO, *, metadata: Mapping[str, str]) -> StoredObject:
        digest = hashlib.sha256()
        byte_size = 0
        with tempfile.NamedTemporaryFile(prefix="clingrounder-artifact-", delete=False) as temporary:
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
                # PRIVACY: bucket names and credentials do not belong in portable manifests.
                uri=content_addressed_object_uri(sha256),
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


def materialize_stored_object(
    store: ArtifactStorePort,
    sha256: str,
    output: str | Path,
    *,
    expected_byte_size: int | None = None,
) -> StoredObject:
    """Atomically restore one CAS object to a seekable local file.

    Archive readers such as ``zipfile`` require a local seekable file, while the canonical
    object may live in S3 or on a differently mounted external disk. This bridge preserves
    content identity without exposing the backend path to downstream stages.
    """

    _validate_digest(sha256)
    if expected_byte_size is not None and expected_byte_size < 0:
        raise ValueError("expected_byte_size must be non-negative")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        observed_sha256, observed_size = _hash_file(destination)
        if observed_sha256 == sha256 and (
            expected_byte_size is None or observed_size == expected_byte_size
        ):
            return StoredObject(
                sha256=sha256,
                uri=content_addressed_object_uri(sha256),
                byte_size=observed_size,
            )
    if not store.exists(sha256):
        raise FileNotFoundError(f"CAS object is missing: {sha256}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with os.fdopen(descriptor, "wb") as target, store.open(sha256) as source:
            # SCALING: stream remote archives in bounded chunks and publish only after
            # integrity checks pass, so interrupted hydration cannot poison a later run.
            while chunk := source.read(_CHUNK_SIZE):
                digest.update(chunk)
                target.write(chunk)
                byte_size += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        if digest.hexdigest() != sha256:
            raise ValueError("materialized object SHA-256 does not match its CAS identity")
        if expected_byte_size is not None and byte_size != expected_byte_size:
            raise ValueError(
                "materialized object byte size does not match release metadata: "
                f"expected {expected_byte_size}, observed {byte_size}"
            )
        os.replace(temporary_path, destination)
        return StoredObject(
            sha256=sha256,
            uri=content_addressed_object_uri(sha256),
            byte_size=byte_size,
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
            byte_size += len(chunk)
    return digest.hexdigest(), byte_size


def _validate_digest(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("Expected a lowercase SHA-256 digest")
