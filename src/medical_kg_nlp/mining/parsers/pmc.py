"""Dispatch PMC OA artifacts to JATS or BioC without changing source policy."""

from __future__ import annotations

from collections.abc import Iterable

from medical_kg_nlp.mining.parsers.json_formats import BiocJsonParser
from medical_kg_nlp.mining.parsers.xml import JatsXmlParser
from medical_kg_nlp.mining.ports import ArtifactStorePort
from medical_kg_nlp.mining.records import MinedDocument, SourceArtifact

__all__ = ["PmcOaParser"]


class PmcOaParser:
    """Select the concrete PMC parser from the checkpointed artifact media type."""

    parser_id = "pmc_oa"
    parser_revision = "1"

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        # SCALING: BioC avoids unreliable package FTP retrieval; pinned JATS archives
        # remain supported when an acquisition plan supplies one explicitly.
        parser = BiocJsonParser() if artifact.media_type == "application/json" else JatsXmlParser()
        yield from parser.parse(artifact, store=store)
