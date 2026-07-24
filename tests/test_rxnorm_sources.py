import json
import subprocess
import sys
import zipfile
from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.rxnorm_sources import (
    RXNORM_CURRENT_FULL_SOURCE_ID,
    RXNORM_CURRENT_PRESCRIBABLE_SOURCE_ID,
    RXNORM_CURRENT_RELEASE_DATE,
    RXNORM_ENRICHMENT_TTYS,
    RXNORM_FULL_FALLBACK_TTYS,
    build_rxnorm_concept_rows,
    parse_rxnorm_rxnconso,
    parse_rxnorm_rxnrel,
    parse_rxnorm_rxnsat,
    profile_rxnorm_release,
    resolve_rxnorm_archive_member_root,
    rxnorm_source_policy,
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
    zip_path = tmp_path / "RxNorm_full_07062026.zip"
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
    assert row["source"] == RXNORM_CURRENT_PRESCRIBABLE_SOURCE_ID
    assert row["rxnorm_status"] == "active"


def test_build_rxnorm_rows_enriches_relations_and_attributes(tmp_path: Path) -> None:
    zip_path = tmp_path / "RxNorm_full_07062026.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "rrf/RXNCONSO.RRF",
            "\n".join(
                [
                    _rxnconso_line("308135", "SCD", "Amlodipine 10 MG Oral Tablet"),
                    _rxnconso_line("17767", "IN", "Amlodipine"),
                    _rxnconso_line("316049", "DF", "Oral Tablet"),
                    _rxnconso_line("999001", "SBD", "Norvasc 10 MG Oral Tablet"),
                    _rxnconso_line("999002", "SBD", "Reactivated Brand Drug"),
                    _rxnconso_line("555001", "BN", "Norvasc"),
                ]
            ),
        )
        archive.writestr(
            "rrf/RXNREL.RRF",
            "\n".join(
                [
                    _rxnrel_line("17767", "308135", rela="HAS_INGREDIENT"),
                    _rxnrel_line("316049", "308135", rela="HAS_DOSE_FORM"),
                    _rxnrel_line("555001", "999001", rela="HAS_TRADENAME"),
                ]
            ),
        )
        archive.writestr(
            "rrf/RXNSAT.RRF",
            "\n".join(
                [
                    _rxnsat_line("308135", "RXN_STRENGTH", "10 MG"),
                    _rxnsat_line("308135", "RXN_ACTIVATED", "20100101"),
                    _rxnsat_line("999001", "RXN_OBSOLETED", "20200101"),
                    _rxnsat_line("999002", "RXN_OBSOLETED", "11/26/2008"),
                    _rxnsat_line("999002", "RXN_ACTIVATED", "01/28/2010"),
                ]
            ),
        )

    rows = build_rxnorm_concept_rows(
        parse_rxnorm_rxnconso(zip_path),
        enrichment_terms=parse_rxnorm_rxnconso(zip_path, allowed_ttys=RXNORM_ENRICHMENT_TTYS),
        relations=parse_rxnorm_rxnrel(zip_path),
        attributes=parse_rxnorm_rxnsat(zip_path),
    )
    by_code = {row["code"]: row for row in rows}

    clinical = by_code["308135"]
    assert clinical["ingredient"] == "Amlodipine"
    assert clinical["dose_form"] == "Oral Tablet"
    assert clinical["strength"] == "RXN_STRENGTH=10 MG"
    assert clinical["rxnorm_activated"] == ["20100101"]
    assert clinical["rxnorm_status"] == "active"
    branded = by_code["999001"]
    assert branded["brand_name"] == "Norvasc 10 MG Oral Tablet"
    assert branded["brand_names"] == ["Norvasc"]
    assert branded["rxnorm_status"] == "inactive"
    assert by_code["999002"]["rxnorm_status"] == "active"


def test_import_rxnorm_dictionary_cli_accepts_prescribable_and_full_sources(tmp_path: Path) -> None:
    prescribable_zip = tmp_path / "RxNorm_prescribable_07062026.zip"
    full_zip = tmp_path / "RxNorm_full_07062026.zip"
    with zipfile.ZipFile(prescribable_zip, "w") as archive:
        archive.writestr("RXNCONSO.RRF", _rxnconso_line("308135", "SCD", "Amlodipine 10 MG Oral Tablet"))
        archive.writestr("RXNREL.RRF", _rxnrel_line("6809", "308135", rel="RO", rela="HAS_INGREDIENT"))
        archive.writestr("RXNSAT.RRF", _rxnsat_line("308135", "TTY", "SCD"))
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
    assert manifest["source_counts"][RXNORM_CURRENT_PRESCRIBABLE_SOURCE_ID] == 1
    assert manifest["source_counts"][RXNORM_CURRENT_FULL_SOURCE_ID] == 1
    assert manifest["source_policy"]["source"]["release_date"] == (
        RXNORM_CURRENT_RELEASE_DATE
    )
    assert manifest["release_profiles"][0]["required_files"]["RXNREL.RRF"] is True
    assert manifest["release_profiles"][0]["rxnrel"]["rela_counts"]["HAS_INGREDIENT"] == 1
    assert manifest["release_profiles"][0]["rxnsat"]["attribute_counts"]["TTY"] == 1
    assert manifest["row_enrichment"]["with_ingredient"] == 2
    assert manifest["row_enrichment"]["with_status"] == 2
    assert metformin.code == "6809"
    assert metformin.code_system == CodeSystem.RXNORM
    assert amlodipine.generic_name == "Amlodipine 10 MG Oral Tablet"


def test_profile_rxnorm_release_counts_conso_rel_and_sat(tmp_path: Path) -> None:
    zip_path = tmp_path / "RxNorm_full_07062026.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "rrf/RXNCONSO.RRF",
            "\n".join(
                [
                    _rxnconso_line("308135", "SCD", "Amlodipine 10 MG Oral Tablet"),
                    _rxnconso_line("308135", "IN", "Amlodipine"),
                    _rxnconso_line("999", "IN", "Suppressed", suppress="Y"),
                ]
            ),
        )
        archive.writestr(
            "rrf/RXNREL.RRF",
            "\n".join(
                [
                    _rxnrel_line("308135", "6809", rel="RO", rela="HAS_INGREDIENT"),
                    _rxnrel_line("308135", "123", sab="MSH"),
                ]
            ),
        )
        archive.writestr(
            "rrf/RXNSAT.RRF",
            "\n".join(
                [
                    _rxnsat_line("308135", "RXN_ACTIVATED", "20200101"),
                    _rxnsat_line("308135", "TTY", "SCD"),
                    _rxnsat_line("308135", "TTY", "MSH", sab="MSH"),
                ]
            ),
        )

    profile = profile_rxnorm_release(zip_path)

    assert profile["required_files"] == {"RXNCONSO.RRF": True, "RXNREL.RRF": True, "RXNSAT.RRF": True}
    assert profile["rxnconso"]["active_concepts"] == 1
    assert profile["rxnconso"]["accepted_concepts"] == 1
    assert profile["rxnconso"]["suppress_counts"]["Y"] == 1
    assert profile["rxnrel"]["rxnorm_rows"] == 1
    assert profile["rxnrel"]["rela_counts"]["HAS_INGREDIENT"] == 1
    assert profile["rxnsat"]["rxnorm_rows"] == 2
    assert profile["rxnsat"]["status_attribute_counts"]["RXN_ACTIVATED"] == 1


def test_bundle_selects_full_and_prescribable_rrf_subtrees(tmp_path: Path) -> None:
    zip_path = tmp_path / "RxNorm_full_07062026.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "rrf/RXNCONSO.RRF",
            "\n".join(
                [
                    _rxnconso_line("100", "IN", "Full Drug"),
                    _rxnconso_line("101", "BN", "Full Brand"),
                ]
            ),
        )
        archive.writestr("rrf/RXNREL.RRF", _rxnrel_line("100", "200"))
        archive.writestr("rrf/RXNSAT.RRF", _rxnsat_line("100", "TTY", "IN"))
        archive.writestr("prescribe/rrf/RXNCONSO.RRF", _rxnconso_line("300", "IN", "Prescribable Drug"))
        archive.writestr("prescribe/rrf/RXNREL.RRF", _rxnrel_line("300", "400"))
        archive.writestr("prescribe/rrf/RXNSAT.RRF", _rxnsat_line("300", "TTY", "IN"))

    full_root = resolve_rxnorm_archive_member_root(zip_path, content="full")
    prescribe_root = resolve_rxnorm_archive_member_root(zip_path, content="prescribable")

    assert full_root == "rrf"
    assert prescribe_root == "prescribe/rrf"
    full_terms = parse_rxnorm_rxnconso(
        zip_path,
        allowed_ttys=RXNORM_FULL_FALLBACK_TTYS,
        archive_member_root=full_root,
    )
    assert [term.rxcui for term in full_terms] == ["100", "101"]
    assert [term.rxcui for term in parse_rxnorm_rxnconso(zip_path, archive_member_root=prescribe_root)] == ["300"]
    full_profile = profile_rxnorm_release(
        zip_path,
        allowed_ttys=RXNORM_FULL_FALLBACK_TTYS,
        archive_member_root=full_root,
    )
    assert full_profile["rxnconso"]["accepted_concepts"] == 2
    assert profile_rxnorm_release(zip_path, archive_member_root=prescribe_root)["rxnconso"]["accepted_concepts"] == 1


def test_default_rxnorm_policy_is_the_current_july_release() -> None:
    source = rxnorm_source_policy()["source"]

    assert source == {
        "fallback_file": "RxNorm_full_07062026.zip",
        "fallback_source_id": "rxnorm_full_2026_07_06",
        "primary_file": "RxNorm_full_07062026.zip",
        "primary_source_id": "rxnorm_prescribable_2026_07_06",
        "release_date": "2026-07-06",
        "source": "NLM RxNorm",
    }


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


def _rxnrel_line(
    rxcui1: str,
    rxcui2: str,
    *,
    rel: str = "RO",
    rela: str = "HAS_TRADENAME",
    sab: str = "RXNORM",
    suppress: str = "N",
) -> str:
    fields = [""] * 16
    fields[0] = rxcui1
    fields[2] = "CUI"
    fields[3] = rel
    fields[4] = rxcui2
    fields[6] = "CUI"
    fields[7] = rela
    fields[10] = sab
    fields[11] = sab
    fields[14] = suppress
    return "|".join(fields) + "|"


def _rxnsat_line(
    rxcui: str,
    attr: str,
    value: str,
    *,
    sab: str = "RXNORM",
    suppress: str = "N",
) -> str:
    fields = [""] * 13
    fields[0] = rxcui
    fields[4] = "CUI"
    fields[8] = attr
    fields[9] = sab
    fields[10] = value
    fields[11] = suppress
    return "|".join(fields) + "|"
