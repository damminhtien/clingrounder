"""Storage, acquisition, labeling, review, and curation extension points."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import BinaryIO, Protocol

from medical_kg_nlp.mining.records import (
    AnnotationProposal,
    CoverageReport,
    DiscoveredArtifact,
    MinedDocument,
    RelationProposal,
    SourceArtifact,
    SourceRequest,
    StoredObject,
)

__all__ = [
    "ArtifactStorePort",
    "CoveragePlannerPort",
    "DeduplicatorPort",
    "DocumentParserPort",
    "ProposalLabelerPort",
    "QualityGatePort",
    "RelationLabelerPort",
    "ReviewBackendPort",
    "SourceConnectorPort",
]


class ArtifactStorePort(Protocol):
    """Persist immutable source bytes behind a content-addressed URI."""

    def put_stream(self, stream: BinaryIO, *, metadata: Mapping[str, str]) -> StoredObject: ...

    def open(self, sha256: str) -> BinaryIO: ...

    def exists(self, sha256: str) -> bool: ...


class SourceConnectorPort(Protocol):
    """Discover source records and materialize them through an artifact store."""

    def discover(self, request: SourceRequest) -> Iterable[DiscoveredArtifact]: ...

    def fetch(
        self,
        artifact: DiscoveredArtifact,
        *,
        store: ArtifactStorePort,
    ) -> SourceArtifact: ...


class DocumentParserPort(Protocol):
    """Parse one materialized artifact into immutable source-text documents."""

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]: ...


class ProposalLabelerPort(Protocol):
    """Produce provenance-bearing proposals without mutating source documents."""

    def propose(self, documents: Sequence[MinedDocument]) -> Iterable[AnnotationProposal]: ...


class RelationLabelerPort(Protocol):
    """Produce source/model relations over an immutable annotation layer."""

    def propose(
        self,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> Iterable[RelationProposal]: ...


class DeduplicatorPort(Protocol):
    """Assign exact and near-duplicate documents to stable groups."""

    def group(self, documents: Sequence[MinedDocument]) -> Mapping[str, str]: ...


class QualityGatePort(Protocol):
    """Return blocking issues for documents and annotations."""

    def validate(
        self,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> Sequence[str]: ...


class ReviewBackendPort(Protocol):
    """Exchange review queues without coupling mining to one annotation UI."""

    def export(
        self,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> str: ...

    def import_reviewed(self, payload: str) -> Sequence[AnnotationProposal]: ...


class CoveragePlannerPort(Protocol):
    """Measure coverage and rank the next records worth reviewing."""

    def report(
        self,
        snapshot_id: str,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> CoverageReport: ...

    def rank(
        self,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> Sequence[str]: ...
