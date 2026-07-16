"""Bounded text/archive parsers for local corpora such as CodiEsp."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable

from medical_kg_nlp.mining.parsers.base import ArtifactParserAdapter
from medical_kg_nlp.mining.ports import ArtifactStorePort
from medical_kg_nlp.mining.records import MinedDocument, SourceArtifact

__all__ = ["CodiEspArchiveParser", "PlainTextParser"]


class PlainTextParser(ArtifactParserAdapter):
    """Parse one UTF-8 text artifact with source-defined language and note type."""

    parser_id = "plain_text"
    parser_revision = "1"

    def __init__(self, *, language: str = "vi", note_type: str = "clinical_text") -> None:
        self.language = language
        self.note_type = note_type

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        text = self.read_artifact(artifact, store).decode("utf-8-sig")
        external_id = artifact.metadata.get("filename", artifact.object.sha256[:16])
        yield self.make_document(
            artifact,
            external_id=external_id,
            text=text,
            language=self.language,
            note_type=self.note_type,
            group_ids=(f"source_record:{external_id}",),
        )


class CodiEspArchiveParser(ArtifactParserAdapter):
    """Parse UTF-8 note members from a CodiEsp ZIP without extracting to disk."""

    parser_id = "codiesp"
    parser_revision = "1"
    max_members = 100_000
    max_member_bytes = 32 * 1024 * 1024

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        payload = self.read_artifact(artifact, store)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = sorted(
                (item for item in archive.infolist() if not item.is_dir() and item.filename.endswith(".txt")),
                key=lambda item: item.filename,
            )
            if len(members) > self.max_members:
                raise ValueError("CodiEsp archive exceeds member count limit")
            if not members:
                raise ValueError("CodiEsp archive contains no .txt documents")
            for member in members:
                if member.file_size > self.max_member_bytes:
                    raise ValueError(f"CodiEsp member {member.filename!r} exceeds size limit")
                # SECURITY: ZipFile.open reads a member directly; paths are never extracted.
                text = archive.read(member).decode("utf-8-sig")
                external_id = member.filename.rsplit("/", 1)[-1].removesuffix(".txt")
                yield self.make_document(
                    artifact,
                    external_id=external_id,
                    text=text,
                    language="es",
                    note_type="clinical_case",
                    group_ids=(f"codiesp_case:{external_id}",),
                    metadata={"archive_member": member.filename},
                )
