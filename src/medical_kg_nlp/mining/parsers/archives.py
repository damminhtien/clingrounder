"""Bounded text/archive parsers for local corpora such as CodiEsp."""

from __future__ import annotations

from collections.abc import Iterable

from medical_kg_nlp.mining.formats.codiesp import read_codiesp_archive
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
    """Parse only Spanish source cases from a bounded CodiEsp archive."""

    parser_id = "codiesp"
    parser_revision = "2"

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        bundle = read_codiesp_archive(
            self.read_artifact(artifact, store),
            include_annotations=False,
        )
        for source_document in bundle.documents:
            external_id = f"{source_document.split}:{source_document.case_id}"
            metadata = {
                "archive_member": source_document.text_member,
                "codiesp_case_id": source_document.case_id,
                "corpus_split": source_document.split,
            }
            if source_document.annotation_member is not None:
                metadata["annotation_member"] = source_document.annotation_member
            yield self.make_document(
                artifact,
                external_id=external_id,
                text=source_document.text,
                language="es",
                note_type="clinical_case",
                group_ids=(f"codiesp_case:{source_document.case_id}",),
                metadata=metadata,
            )
