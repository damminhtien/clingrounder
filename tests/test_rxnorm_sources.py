import json
import subprocess
import sys
import zipfile
from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.rxnorm_sources import (
    RXNORM_FULL_2026_06_01_SOURCE_ID,
    RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID,
    build_rxnorm_concept_rows,
    parse_rxnorm_rxnconso,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_parse_rxnorm_rxnconso_filters_and_builds_dictionary_rows(tmp_path: Path) -> None:
    rxnconso = "\n".join(
        [
            _rxnconso_line("308135", "IN", "Amlodipine"),
            _rxnconso_line("308135", "SCD", "Amlodipine 10 MG Oral Tablet"),
            _rxnconso_line("999", "IN", "Suppressed Drug", suppress="Y"),
            _rxnconso_line("1000", "IN", "External Source Drug", sab="MSH"),
        ]
    )
    zip_path = tmp_path / "RxNorm_full_prescribe_06012026.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("rrf/RXNCONSO.RRF", rxnconso)

    terms = parse_rxnorm_rxnconso(zip_path)
    rows = build_rxnorm_concept_rows(terms)

    assert [term.rxcui for term in terms] == ["308135", "308135"]
    assert len(rows) == 1
    row = rows[0]
    assert row["concept_id"] == "RXNORM:308135"
    assert row["code"] == "308135"
    assert row["code_system"] == CodeSystem.RXNORM.value
    assert row["semantic_type"] == EntityType.DRUG.value
    assert row["canonical_name"] == "Amlodipine 10 MG Oral Tablet"
    assert row["aliases"] == ["Amlodipine"]
    assert row["rxnorm_tty"] == "SCD"
    assert row["source"] == RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID


def test_import_rxnorm_dictionary_cli_accepts_prescribable_and_full_sources(tmp_path: Path) -> None:
    prescribable_zip = tmp_path / "RxNorm_full_prescribe_06012026.zip"
    full_zip = tmp_path / "RxNorm_full_06012026.zip"
    with zipfile.ZipFile(prescribable_zip, "w") as archive:
        archive.writestr("RXNCONSO.RRF", _rxnconso_line("308135", "SCD", "Amlodipine 10 MG Oral Tablet"))
    with zipfile.ZipFile(full_zip, "w") as archive:
        archive.writestr("RXNCONSO.RRF", _rxnconso_line("6809", "IN", "Metformin"))
    output_path = tmp_path / "rxnorm.jsonl"
    manifest_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_rxnorm_dictionary.py",
            "--prescribable-rxnorm",
            str(prescribable_zip),
            "--full-rxnorm",
            str(full_zip),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    store = DictionaryStore.from_jsonl(output_path)
    metformin = store.by_concept_id["RXNORM:6809"]
    amlodipine = store.by_concept_id["RXNORM:308135"]

    assert summary["output"] == str(output_path)
    assert manifest["source_counts"][RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID] == 1
    assert manifest["source_counts"][RXNORM_FULL_2026_06_01_SOURCE_ID] == 1
    assert metformin.code == "6809"
    assert metformin.code_system == CodeSystem.RXNORM
    assert amlodipine.generic_name == "Amlodipine 10 MG Oral Tablet"


def _rxnconso_line(
    rxcui: str,
    tty: str,
    name: str,
    *,
    sab: str = "RXNORM",
    suppress: str = "N",
) -> str:
    fields = [""] * 18
    fields[0] = rxcui
    fields[1] = "ENG"
    fields[2] = "P"
    fields[4] = "PF"
    fields[6] = "Y"
    fields[11] = sab
    fields[12] = tty
    fields[14] = name
    fields[16] = suppress
    return "|".join(fields) + "|"
