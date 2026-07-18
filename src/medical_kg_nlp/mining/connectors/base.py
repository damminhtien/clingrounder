"""Shared acquisition mechanics for registered mining sources."""

from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import closing
from datetime import UTC, datetime
from typing import BinaryIO, Callable, Protocol

from medical_kg_nlp.mining.ports import ArtifactStorePort
from medical_kg_nlp.mining.records import (
    DiscoveredArtifact,
    RedistributionPolicy,
    SourceArtifact,
    SourceRequest,
)
from medical_kg_nlp.mining.registry import LicenseMode, SourceDefinition, VersionPolicy

__all__ = ["BinaryTransportPort", "RegisteredConnectorAdapter"]


class BinaryTransportPort(Protocol):
    """Open one URI as a binary stream without deciding where it is stored."""

    def open(self, uri: str) -> BinaryIO: ...


class _RateLimiter:
    """Serialize request starts so concurrent workers respect a source rate limit."""

    def __init__(self, requests_per_second: float | None) -> None:
        self._interval = 0.0 if requests_per_second is None else 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        if self._interval == 0.0:
            return
        # SCALING: only request admission is serialized; response streaming remains concurrent.
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_start - now)
            if delay:
                time.sleep(delay)
                now = time.monotonic()
            self._next_start = now + self._interval


class RegisteredConnectorAdapter(ABC):
    """Base connector that centralizes version, checksum, and policy handling."""

    connector_revision = "1"

    def __init__(
        self,
        source: SourceDefinition,
        transport: BinaryTransportPort,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.source = source
        self.transport = transport
        self._clock = clock or (lambda: datetime.now(UTC))
        self._rate_limiter = _RateLimiter(source.rate_limit_per_second)

    def discover(self, request: SourceRequest) -> Iterable[DiscoveredArtifact]:
        self._validate_request(request)
        return self._discover(request)

    @abstractmethod
    def _discover(self, request: SourceRequest) -> Iterable[DiscoveredArtifact]:
        """Return deterministic source records without storing their bytes."""

    def fetch(
        self,
        artifact: DiscoveredArtifact,
        *,
        store: ArtifactStorePort,
    ) -> SourceArtifact:
        if artifact.source_id != self.source.id:
            raise ValueError(
                f"Connector {self.source.id!r} cannot fetch source {artifact.source_id!r}"
            )
        self._rate_limiter.wait()
        with closing(self.transport.open(artifact.uri)) as stream:
            stored = store.put_stream(
                stream,
                metadata={
                    "source_id": artifact.source_id,
                    "source_version": artifact.source_version,
                    "source_uri": artifact.uri,
                    **artifact.metadata,
                },
            )
        if artifact.expected_sha256 is not None and stored.sha256 != artifact.expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {artifact.uri}: "
                f"expected {artifact.expected_sha256}, received {stored.sha256}"
            )

        license_id = self._resolve_license(artifact)
        redistribution = _metadata_redistribution(
            artifact.metadata.get("redistribution"),
            default=self.source.redistribution,
        )
        artifact_id_seed = (
            f"{artifact.source_id}\0{artifact.source_version}\0{stored.sha256}".encode()
        )
        artifact_id = f"{artifact.source_id}:{hashlib.sha256(artifact_id_seed).hexdigest()[:24]}"
        return SourceArtifact(
            artifact_id=artifact_id,
            source_id=artifact.source_id,
            source_version=artifact.source_version,
            source_uri=artifact.uri,
            object=stored,
            media_type=artifact.media_type,
            license_id=license_id,
            access_class=self.source.access_class,
            redistribution=redistribution,
            hosted_processing_allowed=self.source.hosted_processing_allowed,
            retrieved_at=self._clock().astimezone(UTC).isoformat(),
            metadata=dict(artifact.metadata),
        )

    def open_uri(self, uri: str) -> BinaryIO:
        """Open discovery metadata through the same source rate limiter."""

        self._rate_limiter.wait()
        return self.transport.open(uri)

    def _validate_request(self, request: SourceRequest) -> None:
        if request.source_id != self.source.id:
            raise ValueError(
                f"Connector {self.source.id!r} received request for {request.source_id!r}"
            )
        if (
            self.source.version_policy is VersionPolicy.PINNED
            and request.source_version != self.source.version
        ):
            raise ValueError(
                f"Pinned source {self.source.id!r} requires version {self.source.version!r}"
            )

    def _resolve_license(self, artifact: DiscoveredArtifact) -> str:
        if self.source.license_mode is LicenseMode.FIXED:
            return self.source.license_id
        license_id = artifact.metadata.get("license_id", "").strip()
        if not license_id:
            # LICENSE: per-artifact sources are blocked until the actual license is known.
            raise ValueError(f"Artifact {artifact.uri!r} has no per-artifact license")
        return license_id


def _metadata_redistribution(
    value: str | None,
    *,
    default: RedistributionPolicy,
) -> RedistributionPolicy:
    if value is None:
        return default
    try:
        return RedistributionPolicy(value)
    except ValueError as error:
        raise ValueError(f"Unknown artifact redistribution policy {value!r}") from error
