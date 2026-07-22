"""Exact RxNorm NDC indexing and DailyMed product-linking contracts."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from medical_kg_nlp.mining.io import write_jsonl
from medical_kg_nlp.mining.mappings.dailymed_product_rxnorm import (
    link_dailymed_products_to_rxnorm,
)
from medical_kg_nlp.mining.mappings.rxnorm_ndc import (
    RxNormNdcRepository,
    compile_rxnorm_ndc_index,
    normalize_ndc11,
    normalize_ndc_product_prefix,
)
from medical_kg_nlp.mining.records import (
    AccessClass,
    MinedDocument,
    RedistributionPolicy,
)
from medical_kg_nlp.utils.hashing import sha256_file


def test_ndc_normalization_preserves_product_identity() -> None:
    assert normalize_ndc_product_prefix("1234-5678") == "012345678"
    assert normalize_ndc_product_prefix("12345-678") == "123450678"
    assert normalize_ndc_product_prefix("12345-6789") == "123456789"
    assert normalize_ndc11("1234-5678-90") == "01234567890"
    assert normalize_ndc11("12345-678-90") == "12345067890"
    assert normalize_ndc11("12345-6789-0") == "12345678900"

    with pytest.raises(ValueError, match="Unsupported NDC product"):
        normalize_ndc_product_prefix("123-45")


def test_rxnorm_ndc_compiler_is_deduplicated_pinned_and_queryable(
    tmp_path: Path,
) -> None:
    archive = _rxnorm_archive(
        tmp_path,
        [
            _rxnsat_line("100", "01234567890"),
            _rxnsat_line("100", "01234567890"),
            _rxnsat_line("101", "01234567891"),
            _rxnsat_line("999", "not-an-ndc"),
            _rxnsat_line("888", "01234567892", suppress="Y"),
        ],
    )
    source_sha256 = sha256_file(archive)
    output = tmp_path / "ndc.jsonl"
    index = tmp_path / "ndc.sqlite3"
    report_path = tmp_path / "report.json"

    report = compile_rxnorm_ndc_index(
        archive,
        source_version="2026-07-06",
        expected_source_sha256=source_sha256,
        output_path=output,
        index_path=index,
        report_path=report_path,
    )

    assert report["active_ndc_row_count"] == 4
    assert report["invalid_ndc_row_count"] == 1
    assert report["unique_row_count"] == 2
    assert report["duplicate_active_row_count"] == 2
    repository = RxNormNdcRepository(
        index, expected_source_sha256=source_sha256
    )
    assert repository.lookup("1234-5678") == ("100", "101")
    assert repository.lookup("01234567890") == ("100",)
    repository.close()

    with pytest.raises(ValueError, match="source fingerprint changed"):
        RxNormNdcRepository(index, expected_source_sha256="0" * 64)
    with pytest.raises(ValueError, match="source fingerprint changed"):
        compile_rxnorm_ndc_index(
            archive,
            source_version="2026-07-06",
            expected_source_sha256="0" * 64,
            output_path=output,
            index_path=index,
            report_path=report_path,
        )


def test_dailymed_product_linker_requires_unique_two_source_agreement(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents.jsonl"
    write_jsonl(
        documents,
        (
            _document("doc-a", set_id="set-a", ndc="1234-5678"),
            _document("doc-b", set_id="set-b", ndc="54321-876"),
            _document("doc-c", set_id="set-c", ndc="11111-222"),
        ),
    )
    mapping_index = tmp_path / "mapping.sqlite3"
    _mapping_index(
        mapping_index,
        {
            ("set-a", "1"): {"100": "Example 10 MG Oral Tablet"},
            ("set-b", "1"): {
                "200": "Ambiguous 10 MG Tablet",
                "201": "Ambiguous 20 MG Tablet",
            },
            ("set-c", "1"): {"300": "Disagreement Tablet"},
        },
    )
    ndc_archive = _rxnorm_archive(
        tmp_path,
        [
            _rxnsat_line("100", "01234567890"),
            _rxnsat_line("200", "54321087601"),
            _rxnsat_line("201", "54321087602"),
            _rxnsat_line("999", "11111022201"),
        ],
    )
    ndc_index = tmp_path / "ndc.sqlite3"
    compile_rxnorm_ndc_index(
        ndc_archive,
        source_version="2026-07-06",
        expected_source_sha256=sha256_file(ndc_archive),
        output_path=tmp_path / "ndc.jsonl",
        index_path=ndc_index,
        report_path=tmp_path / "ndc-report.json",
    )
    links = tmp_path / "links.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    report_path = tmp_path / "link-report.json"

    report = link_dailymed_products_to_rxnorm(
        documents,
        dailymed_mapping_index_path=mapping_index,
        rxnorm_ndc_index_path=ndc_index,
        links_path=links,
        decisions_path=decisions,
        report_path=report_path,
    )

    link_rows = _jsonl(links)
    decision_rows = _jsonl(decisions)
    assert len(link_rows) == 1
    assert link_rows[0]["document_id"] == "doc-a"
    assert link_rows[0]["rxcui"] == "100"
    assert link_rows[0]["evidence"] == "exact_set_version_ndc_intersection"
    assert report["status_counts"] == {
        "ambiguous_intersection": 1,
        "exact_unique_intersection": 1,
        "source_disagreement": 1,
    }
    assert {row["status"] for row in decision_rows} == {
        "ambiguous_intersection",
        "exact_unique_intersection",
        "source_disagreement",
    }
    assert str(tmp_path) not in report_path.read_text(encoding="utf-8")


def _document(
    document_id: str,
    *,
    set_id: str,
    ndc: str,
) -> dict[str, object]:
    text = "Product: Example\nNDC: " + ndc
    document = MinedDocument(
        document_id=document_id,
        text=text,
        language="en",
        note_type="structured_medication_record",
        source_artifact_id="dailymed:fixture",
        access_class=AccessClass.OPEN_WITH_TERMS,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        metadata={
            "dailymed_set_id": set_id,
            "dailymed_spl_version": "1",
            "dailymed_source_version": "fixture-release",
            "spl_ndc": ndc,
            "spl_fields": json.dumps(
                [
                    {
                        "span": [9, 16],
                        "text": "Example",
                        "source_label": "SPL_PRODUCT_NAME",
                        "role": "product",
                    }
                ],
                separators=(",", ":"),
            ),
        },
    )
    return document.to_dict()


def _mapping_index(
    path: Path,
    values: dict[tuple[str, str], dict[str, str]],
) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
    )
    connection.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        (
            ("schema_version", "dailymed-rxnorm-mapping.v2"),
            ("source_version", "mapping-fixture"),
            ("source_sha256", "a" * 64),
        ),
    )
    connection.execute(
        """
        CREATE TABLE mapping_rows (
            set_id TEXT NOT NULL,
            spl_version TEXT NOT NULL,
            rxcui TEXT NOT NULL,
            rxstring TEXT NOT NULL,
            normalized TEXT NOT NULL,
            rxtty TEXT NOT NULL,
            PRIMARY KEY (set_id, spl_version, rxcui, rxstring, rxtty)
        ) WITHOUT ROWID
        """
    )
    for (set_id, version), concepts in values.items():
        for rxcui, text in concepts.items():
            connection.execute(
                "INSERT INTO mapping_rows VALUES (?, ?, ?, ?, ?, ?)",
                (set_id, version, rxcui, text, text.casefold(), "SCD"),
            )
    connection.commit()
    connection.close()


def _rxnorm_archive(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / f"rxnorm-{len(list(tmp_path.glob('rxnorm-*.zip')))}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rrf/RXNSAT.RRF", "\n".join(lines))
    return path


def _rxnsat_line(rxcui: str, ndc: str, *, suppress: str = "N") -> str:
    fields = [""] * 13
    fields[0] = rxcui
    fields[8] = "NDC"
    fields[9] = "RXNORM"
    fields[10] = ndc
    fields[11] = suppress
    return "|".join(fields) + "|"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
