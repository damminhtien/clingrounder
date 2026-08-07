import json
import zipfile
from pathlib import Path

from clingrounder.dictionaries.source_audit import (
    build_source_audit_report,
    false_positive_blocklist_candidates,
    file_fingerprints,
    profile_dictionary,
    write_source_audit_report,
)


def test_file_fingerprints_records_md5_sha256_and_size(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    path.write_text("abc", encoding="utf-8")

    result = file_fingerprints(path)

    assert result["exists"] is True
    assert result["size_bytes"] == 3
    assert result["md5"] == "900150983cd24fb0d6963f7d28e17f72"
    assert result["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_profile_dictionary_reports_ambiguous_and_broad_aliases(tmp_path: Path) -> None:
    dictionary = tmp_path / "dict.jsonl"
    _write_jsonl(
        dictionary,
        [
            {
                "concept_id": "ICD10:E11",
                "code": "E11",
                "code_system": "ICD-10",
                "canonical_name": "Diabetes",
                "semantic_type": "DISEASE",
                "aliases": ["DM"],
                "source": "seed",
            },
            {
                "concept_id": "LOCAL:SYMPTOM_DM",
                "code": "SYMPTOM_DM",
                "code_system": "LOCAL",
                "canonical_name": "DM",
                "semantic_type": "SYMPTOM",
                "aliases": ["DM"],
                "source": "seed",
            },
            {
                "concept_id": "RXNORM:61",
                "code": "61",
                "code_system": "RxNorm",
                "canonical_name": "alanine",
                "semantic_type": "DRUG",
                "source": "rxnorm_prescribable_2026_06_01",
            },
        ],
    )

    profile = profile_dictionary(dictionary, known_source_ids={"seed", "rxnorm_prescribable_2026_06_01"})

    assert profile["row_count"] == 3
    assert profile["ambiguous_alias_count"] == 1
    assert profile["ambiguous_aliases"][0]["normalized_alias"] == "dm"
    assert any(row["reason"] == "too_short" for row in profile["broad_aliases"])
    assert any(row["reason"] == "lab_or_metabolite_rxnorm_alias_requires_drug_context" for row in profile["broad_aliases"])


def test_false_positive_blocklist_candidates_prioritize_context_gated_drug_aliases(tmp_path: Path) -> None:
    dictionary = tmp_path / "dict.jsonl"
    _write_jsonl(
        dictionary,
        [
            {
                "concept_id": "ICD10:J18.9",
                "code": "J18.9",
                "code_system": "ICD-10",
                "canonical_name": "Pneumonia",
                "semantic_type": "DISEASE",
                "aliases": ["HF"],
                "source": "seed",
            },
            {
                "concept_id": "RXNORM:4850",
                "code": "4850",
                "code_system": "RxNorm",
                "canonical_name": "lactate",
                "semantic_type": "DRUG",
                "source": "rxnorm_prescribable_2026_06_01",
            },
        ],
    )
    profile = profile_dictionary(dictionary, known_source_ids={"seed", "rxnorm_prescribable_2026_06_01"})

    rows = false_positive_blocklist_candidates([profile])

    assert rows[0]["concept_id"] == "RXNORM:4850"
    assert rows[0]["severity"] == "high"
    assert rows[0]["recommended_action"] == "block_unless_explicit_drug_context"
    assert any(row["reason"] == "too_short" for row in rows)


def test_build_source_audit_report_writes_artifacts_and_rxnorm_profile(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    raw = tmp_path / "raw.txt"
    raw.write_text("source", encoding="utf-8")
    registry.write_text(
        """
resources:
  - id: seed
    name: Seed
    category: curation
    access: local
    url: docs/dictionaries.md
    version: local
    license: project
    use: runtime_seed
    local_files:
      - role: raw
        path: REPLACE_RAW
        required: true
  - id: rxnorm_prescribable_2026_06_01
    name: RxNorm
    category: terminology
    access: open_with_terms
    url: https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html
    version: "2026-06-01"
    release_date: "2026-06-01"
    license: nlm_rxnorm_terms
    use: drug_codes
""".replace("REPLACE_RAW", str(raw)),
        encoding="utf-8",
    )
    versions = tmp_path / "versions.json"
    versions.write_text(
        json.dumps(
            {
                "icd10_vn": {"source_id": "seed"},
                "rxnorm": {"primary_source_id": "rxnorm_prescribable_2026_06_01"},
            }
        ),
        encoding="utf-8",
    )
    dictionary = tmp_path / "dict.jsonl"
    _write_jsonl(
        dictionary,
        [
            {
                "concept_id": "RXNORM:308135",
                "code": "308135",
                "code_system": "RxNorm",
                "canonical_name": "Amlodipine 10 MG Oral Tablet",
                "semantic_type": "DRUG",
                "source": "rxnorm_prescribable_2026_06_01",
            },
            {
                "concept_id": "RXNORM:61",
                "code": "61",
                "code_system": "RxNorm",
                "canonical_name": "alanine",
                "semantic_type": "DRUG",
                "source": "rxnorm_prescribable_2026_06_01",
            }
        ],
    )
    rxnorm_zip = tmp_path / "rxnorm.zip"
    with zipfile.ZipFile(rxnorm_zip, "w") as archive:
        archive.writestr(
            "rrf/RXNCONSO.RRF",
            _rxnconso_line("308135", "SCD", "Amlodipine 10 MG Oral Tablet")
            + "\n"
            + _rxnconso_line("42844", "BN", "Percocet"),
        )
        archive.writestr("rrf/RXNREL.RRF", _rxnrel_line("308135", "6809"))
        archive.writestr("rrf/RXNSAT.RRF", _rxnsat_line("308135", "TTY", "SCD"))

    report = build_source_audit_report(
        registry_path=registry,
        standard_versions_path=versions,
        dictionary_paths=[dictionary],
        rxnorm_release_paths=[{"path": rxnorm_zip, "archive_member_root": "rrf", "content": "full"}],
    )
    output_dir = tmp_path / "audit"
    write_source_audit_report(report, output_dir)

    assert report["summary"]["missing_required_file_count"] == 0
    assert report["summary"]["false_positive_blocklist_count"] == 1
    resources = {row["id"]: row for row in report["registry"]["resources"]}
    assert resources["rxnorm_prescribable_2026_06_01"]["license"] == "nlm_rxnorm_terms"
    assert resources["seed"]["local_files"][0]["path"] == str(raw)
    assert report["rxnorm_release_profiles"][0]["rxnrel"]["active_rows"] == 1
    assert report["rxnorm_release_profiles"][0]["archive_member_root"] == "rrf"
    assert report["rxnorm_release_profiles"][0]["content"] == "full"
    assert report["rxnorm_release_profiles"][0]["rxnconso"]["accepted_concepts"] == 2
    assert report["false_positive_blocklist"][0]["concept_id"] == "RXNORM:61"
    assert (output_dir / "source_manifest.json").exists()
    assert (output_dir / "dictionary_coverage.md").exists()
    assert (output_dir / "manual_review_queue.jsonl").exists()
    assert (output_dir / "false_positive_blocklist.jsonl").exists()
    assert (output_dir / "false_positive_blocklist.md").exists()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _rxnconso_line(rxcui: str, tty: str, name: str) -> str:
    fields = [""] * 18
    fields[0] = rxcui
    fields[1] = "ENG"
    fields[6] = "Y"
    fields[11] = "RXNORM"
    fields[12] = tty
    fields[14] = name
    fields[16] = "N"
    return "|".join(fields) + "|"


def _rxnrel_line(rxcui1: str, rxcui2: str) -> str:
    fields = [""] * 16
    fields[0] = rxcui1
    fields[3] = "RO"
    fields[4] = rxcui2
    fields[7] = "HAS_INGREDIENT"
    fields[10] = "RXNORM"
    fields[14] = "N"
    return "|".join(fields) + "|"


def _rxnsat_line(rxcui: str, attr: str, value: str) -> str:
    fields = [""] * 13
    fields[0] = rxcui
    fields[8] = attr
    fields[9] = "RXNORM"
    fields[10] = value
    fields[11] = "N"
    return "|".join(fields) + "|"
