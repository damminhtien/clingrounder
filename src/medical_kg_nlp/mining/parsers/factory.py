"""Parser composition based on registry v2 parser identifiers."""

from __future__ import annotations

from collections.abc import Callable

from medical_kg_nlp.mining.parsers.archives import CodiEspArchiveParser, PlainTextParser
from medical_kg_nlp.mining.parsers.json_formats import (
    BiocJsonParser,
    ClinicalTrialsJsonParser,
    FhirBundleParser,
)
from medical_kg_nlp.mining.parsers.xml import JatsXmlParser, SplXmlParser
from medical_kg_nlp.mining.ports import DocumentParserPort
from medical_kg_nlp.mining.registry import SourceDefinition

__all__ = ["parser_from_definition"]


def parser_from_definition(source: SourceDefinition) -> DocumentParserPort:
    """Instantiate implemented parsers and reject silent format assumptions."""

    parser_factories: dict[str, Callable[[], DocumentParserPort]] = {
        "jats_xml": JatsXmlParser,
        "spl_xml": SplXmlParser,
        "clinicaltrials_json": ClinicalTrialsJsonParser,
        "fhir_bundle": FhirBundleParser,
        "bioc": BiocJsonParser,
        "codiesp": CodiEspArchiveParser,
    }
    parser_factory = parser_factories.get(source.parser)
    if parser_factory is not None:
        return parser_factory()
    if source.parser in {"vietnamese_guideline", "mimic_iv_note", "n2c2"}:
        return PlainTextParser(language="vi" if source.parser == "vietnamese_guideline" else "en")
    raise ValueError(
        f"Source {source.id!r} declares parser {source.parser!r}, which is not implemented"
    )
