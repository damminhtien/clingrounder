"""Shared document construction and bounded artifact-reading helpers."""

from __future__ import annotations

import hashlib
from typing import BinaryIO

from medical_kg_nlp.mining.ports import ArtifactStorePort
from medical_kg_nlp.mining.records import MinedDocument, SourceArtifact

__all__ = ["ArtifactParserAdapter"]


class ArtifactParserAdapter:
    """Base parser preserving policy metadata on every derived document."""

    parser_id = "base"
    parser_revision = "1"
    max_artifact_bytes = 512 * 1024 * 1024

    def read_artifact(self, artifact: SourceArtifact, store: ArtifactStorePort) -> bytes:
        with store.open(artifact.object.sha256) as stream:
            return _read_bounded(stream, self.max_artifact_bytes)

    def make_document(
        self,
        artifact: SourceArtifact,
        *,
        external_id: str,
        text: str,
        language: str,
        note_type: str,
        group_ids: tuple[str, ...] = (),
        metadata: dict[str, str] | None = None,
    ) -> MinedDocument:
        if not text.strip():
            raise ValueError(f"Parser {self.parser_id!r} produced an empty document")
        identity = (
            f"{artifact.object.sha256}\0{self.parser_id}\0{self.parser_revision}\0{external_id}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        # INVARIANT: parsed text is frozen here; later normalization must create a child document.
        return MinedDocument(
            document_id=f"{artifact.source_id}:{digest}",
            text=text,
            language=language,
            note_type=note_type,
            source_artifact_id=artifact.artifact_id,
            access_class=artifact.access_class,
            redistribution=artifact.redistribution,
            hosted_processing_allowed=artifact.hosted_processing_allowed,
            group_ids=group_ids,
            metadata={
                "external_id": external_id,
                "parser_id": self.parser_id,
                "parser_revision": self.parser_revision,
                **(metadata or {}),
            },
        )


def _read_bounded(stream: BinaryIO, limit: int) -> bytes:
    payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"Artifact exceeds parser limit of {limit} bytes")
    return payload
