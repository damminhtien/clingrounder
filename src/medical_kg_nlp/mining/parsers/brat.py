"""BRAT archive parser preserving source text and annotation-member provenance."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from medical_kg_nlp.mining.formats.brat import read_brat_archive
from medical_kg_nlp.mining.parsers.base import ArtifactParserAdapter
from medical_kg_nlp.mining.ports import ArtifactStorePort
from medical_kg_nlp.mining.records import MinedDocument, SourceArtifact

__all__ = ["BratArchiveParser"]


class BratArchiveParser(ArtifactParserAdapter):
    """Parse paired BRAT members while leaving source labels to a labeler adapter."""

    parser_id = "brat"
    parser_revision = "2"

    def __init__(self, *, language: str = "und", note_type: str = "annotated_text") -> None:
        if not language.strip() or not note_type.strip():
            raise ValueError("BRAT language and note_type must be non-empty")
        self.language = language
        self.note_type = note_type

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        bundles = read_brat_archive(self.read_artifact(artifact, store))
        for bundle in bundles:
            text_sha256 = hashlib.sha256(bundle.text.encode("utf-8")).hexdigest()
            external_id = bundle.text_member.removesuffix(".txt")
            yield self.make_document(
                artifact,
                external_id=external_id,
                text=bundle.text,
                language=self.language,
                note_type=self.note_type,
                # SCALING: exact duplicates stay in one split even before near-dedup runs.
                group_ids=(f"exact_text:{text_sha256}",),
                metadata={
                    "archive_member": bundle.text_member,
                    "annotation_member": bundle.annotation_member,
                    "annotation_format": "brat",
                    "newline_normalization": bundle.newline_normalization,
                },
            )
