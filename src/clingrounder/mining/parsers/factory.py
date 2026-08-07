"""Parser composition based on registry v2 parser identifiers."""

from __future__ import annotations

from collections.abc import Callable

from clingrounder.mining.parsers.brat import BratArchiveParser
from clingrounder.mining.parsers.archives import (
    CodiEspArchiveParser,
    PlainTextArchiveParser,
    PlainTextParser,
)
from clingrounder.mining.parsers.json_formats import (
    BiocJsonParser,
    ClinicalTrialsJsonParser,
    FhirBundleParser,
)
from clingrounder.mining.parsers.pmc import PmcOaParser
from clingrounder.mining.parsers.vietmed_ner import VietMedNerParquetParser
from clingrounder.mining.parsers.xml import JatsXmlParser, SplXmlParser
from clingrounder.mining.ports import DocumentParserPort
from clingrounder.mining.registry import SourceDefinition

__all__ = ["parser_from_definition"]


def parser_from_definition(source: SourceDefinition) -> DocumentParserPort:
    """Instantiate implemented parsers and reject silent format assumptions."""

    parser_factories: dict[str, Callable[[], DocumentParserPort]] = {
        "jats_xml": JatsXmlParser,
        "pmc_oa": PmcOaParser,
        "spl_xml": SplXmlParser,
        "clinicaltrials_json": ClinicalTrialsJsonParser,
        "fhir_bundle": FhirBundleParser,
        "bioc": BiocJsonParser,
        "codiesp": CodiEspArchiveParser,
        "vietmed_ner_parquet": VietMedNerParquetParser,
    }
    parser_factory = parser_factories.get(source.parser)
    if parser_factory is not None:
        return parser_factory()
    if source.parser == "brat":
        return BratArchiveParser(
            language=source.parser_options.get("language", "und"),
            note_type=source.parser_options.get("note_type", "annotated_text"),
        )
    if source.parser == "plain_text_archive":
        options = source.parser_options
        return PlainTextArchiveParser(
            language=options.get("language", "vi"),
            note_type=options.get("note_type", "clinical_text"),
            require_numeric_ids=_boolean_option(
                options,
                "require_numeric_ids",
                default=True,
            ),
            max_members=_integer_option(options, "max_members", default=10_000),
            max_member_bytes=_integer_option(
                options,
                "max_member_bytes",
                default=8 * 1024 * 1024,
            ),
            max_total_uncompressed_bytes=_integer_option(
                options,
                "max_total_uncompressed_bytes",
                default=256 * 1024 * 1024,
            ),
            max_compression_ratio=_float_option(
                options,
                "max_compression_ratio",
                default=200.0,
            ),
        )
    if source.parser in {"vietnamese_guideline", "mimic_iv_note", "n2c2"}:
        return PlainTextParser(language="vi" if source.parser == "vietnamese_guideline" else "en")
    raise ValueError(
        f"Source {source.id!r} declares parser {source.parser!r}, which is not implemented"
    )


def _integer_option(options: dict[str, str], key: str, *, default: int) -> int:
    raw = options.get(key)
    return default if raw is None else int(raw)


def _float_option(options: dict[str, str], key: str, *, default: float) -> float:
    raw = options.get(key)
    return default if raw is None else float(raw)


def _boolean_option(options: dict[str, str], key: str, *, default: bool) -> bool:
    raw = options.get(key)
    if raw is None:
        return default
    normalized = raw.casefold()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Parser option {key!r} must be true or false")
    return normalized == "true"
