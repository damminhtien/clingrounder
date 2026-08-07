import json
import subprocess
import sys
import zipfile
from pathlib import Path

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.dictionaries.icd10_sources import (
    ICD10_VN_TT06_2026_SOURCE_ID,
    build_icd10_concept_rows,
    icd10_chapter_for_code,
    icd10_source_policy,
    load_icd10_vietnamese_overlays,
    parse_cdc_icd10cm_descriptions,
    parse_cdc_icd10cm_tabular_xml,
    parse_icd10_vn_tt06_table,
    parse_who_icd10_claml,
    write_icd10_concept_rows,
)
from clingrounder.schema.types import CodeSystem, EntityType


WHO_CLAML_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<ClaML xmlns="urn:who:icd10:test">
  <Class kind="block" code="E10-E14">
    <Rubric kind="preferred"><Label>Diabetes mellitus</Label></Rubric>
  </Class>
  <Class kind="category" code="E11">
    <SuperClass code="E10-E14"/>
    <Rubric kind="preferred"><Label>Type 2 diabetes mellitus</Label></Rubric>
    <Rubric kind="inclusion"><Label>Non-insulin-dependent diabetes mellitus</Label></Rubric>
  </Class>
  <Class kind="category" code="J45">
    <SuperClass code="J40-J47"/>
    <Rubric kind="preferred"><Label>Asthma</Label></Rubric>
  </Class>
</ClaML>
"""

CDC_TABULAR_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<ICD10CM.tabular>
  <chapter>
    <diag>
      <name>A00</name>
      <desc>Cholera</desc>
      <diag>
        <name>A00.0</name>
        <desc>Cholera due to Vibrio cholerae 01, biovar cholerae</desc>
        <inclusionTerm>
          <note>Classical cholera</note>
        </inclusionTerm>
      </diag>
    </diag>
    <diag>
      <name>E11.9</name>
      <desc>Type 2 diabetes mellitus without complications</desc>
    </diag>
  </chapter>
</ICD10CM.tabular>
"""


def test_parse_who_icd10_claml_filters_blocks_and_keeps_hierarchy(tmp_path: Path) -> None:
    claml_path = tmp_path / "icd10.xml"
    claml_path.write_text(WHO_CLAML_FIXTURE, encoding="utf-8")

    concepts = parse_who_icd10_claml(claml_path)

    assert [concept.code for concept in concepts] == ["E11", "J45"]
    diabetes = concepts[0]
    assert diabetes.official_name_en == "Type 2 diabetes mellitus"
    assert diabetes.parent_code == "E10-E14"
    assert diabetes.aliases == ("Non-insulin-dependent diabetes mellitus",)


def test_parse_cdc_icd10cm_descriptions_formats_undotted_codes(tmp_path: Path) -> None:
    cdc_path = tmp_path / "cdc_descriptions.txt"
    cdc_path.write_text(
        "\n".join(
            [
                "E119 Type 2 diabetes mellitus without complications",
                "I2510 Atherosclerotic heart disease of native coronary artery without angina pectoris",
                "J44.9 Chronic obstructive pulmonary disease, unspecified",
            ]
        ),
        encoding="utf-8",
    )

    concepts = parse_cdc_icd10cm_descriptions(cdc_path)

    by_code = {concept.code: concept for concept in concepts}
    assert by_code["E11.9"].parent_code == "E11"
    assert by_code["I25.10"].official_name_en.startswith("Atherosclerotic heart disease")
    assert by_code["J44.9"].parent_code == "J44"


def test_parse_cdc_icd10cm_tabular_xml_keeps_nested_parent_and_inclusions(
    tmp_path: Path,
) -> None:
    cdc_xml_path = tmp_path / "icd10c-tabular.xml"
    cdc_xml_path.write_text(CDC_TABULAR_FIXTURE, encoding="utf-8")

    concepts = parse_cdc_icd10cm_tabular_xml(cdc_xml_path)

    by_code = {concept.code: concept for concept in concepts}
    assert by_code["A00.0"].parent_code == "A00"
    assert by_code["A00.0"].aliases == ("Classical cholera",)
    assert by_code["E11.9"].official_name_en == "Type 2 diabetes mellitus without complications"


def test_parse_icd10_vn_tt06_structured_extract_keeps_vietnamese_primary_name(
    tmp_path: Path,
) -> None:
    tt06_path = tmp_path / "tt06.jsonl"
    tt06_path.write_text(
        json.dumps(
            {
                "ma_benh": "E11",
                "ten_benh": "Đái tháo đường không phụ thuộc insulin",
                "official_name_en": "Type 2 diabetes mellitus",
                "aliases": ["đái tháo đường type 2", "tiểu đường type 2"],
                "parent_code": "E10-E14",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    concepts = parse_icd10_vn_tt06_table(tt06_path)

    assert len(concepts) == 1
    concept = concepts[0]
    assert concept.source_id == ICD10_VN_TT06_2026_SOURCE_ID
    assert concept.code == "E11"
    assert concept.official_name_vi == "Đái tháo đường không phụ thuộc insulin"
    assert concept.official_name_en == "Type 2 diabetes mellitus"
    assert concept.parent_code == "E10-E14"
    assert concept.aliases == ("đái tháo đường type 2", "tiểu đường type 2")


def test_build_icd10_concept_rows_prefers_tt06_over_cdc_for_phase1(tmp_path: Path) -> None:
    cdc_path = tmp_path / "cdc.txt"
    cdc_path.write_text("E11 Type 2 diabetes mellitus\n", encoding="utf-8")
    tt06_path = tmp_path / "tt06.jsonl"
    tt06_path.write_text(
        json.dumps(
            {
                "code": "E11",
                "official_name_vi": "Đái tháo đường không phụ thuộc insulin",
                "official_name_en": "Type 2 diabetes mellitus",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_icd10_concept_rows(
        [
            *parse_cdc_icd10cm_descriptions(cdc_path),
            *parse_icd10_vn_tt06_table(tt06_path),
        ]
    )

    diabetes = rows[0]
    assert diabetes["code"] == "E11"
    assert diabetes["canonical_name"] == "Đái tháo đường không phụ thuộc insulin"
    assert diabetes["official_name_vi"] == "Đái tháo đường không phụ thuộc insulin"
    assert diabetes["source"] == ICD10_VN_TT06_2026_SOURCE_ID
    assert diabetes["source_ids"] == ["cdc_icd10cm_2026", ICD10_VN_TT06_2026_SOURCE_ID]
    assert diabetes["icd10_chapter"] == "IV"
    assert diabetes["icd10_chapter_range"] == "E00-E90"


def test_build_icd10_concept_rows_merges_sources_and_vietnamese_aliases(
    tmp_path: Path,
) -> None:
    claml_path = tmp_path / "icd10.xml"
    claml_path.write_text(WHO_CLAML_FIXTURE, encoding="utf-8")
    cdc_path = tmp_path / "cdc_descriptions.txt"
    cdc_path.write_text("E11 Type 2 diabetes mellitus\nE119 Type 2 diabetes mellitus without complications\n", encoding="utf-8")
    alias_path = tmp_path / "vn_aliases.jsonl"
    alias_path.write_text(
        json.dumps(
            {
                "target_concept_id": "ICD10:E11",
                "alias": "tiểu đường",
                "canonical": "đái tháo đường",
                "official_name_vi": "Đái tháo đường type 2",
                "semantic_type": "DISEASE",
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {
                "target_concept_id": "LOCAL:SYMPTOM_COUGH",
                "alias": "ho",
                "semantic_type": "SYMPTOM",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_icd10_concept_rows(
        [
            *parse_who_icd10_claml(claml_path),
            *parse_cdc_icd10cm_descriptions(cdc_path),
        ],
        load_icd10_vietnamese_overlays(alias_path),
    )

    by_code = {row["code"]: row for row in rows}
    diabetes = by_code["E11"]
    assert diabetes["concept_id"] == "ICD10:E11"
    assert diabetes["code_system"] == CodeSystem.ICD10.value
    assert diabetes["semantic_type"] == EntityType.DISEASE.value
    assert diabetes["canonical_name"] == "Type 2 diabetes mellitus"
    assert diabetes["official_name_vi"] == "Đái tháo đường type 2"
    assert diabetes["aliases"] == [
        "Non-insulin-dependent diabetes mellitus",
        "tiểu đường",
        "đái tháo đường",
        "Đái tháo đường type 2",
    ]
    assert diabetes["source"] == "who_icd10_2019"
    assert diabetes["source_ids"] == ["cdc_icd10cm_2026", "icd_kcb_vn", "who_icd10_2019"]
    assert by_code["E11.9"]["parent_code"] == "E11"

    output_path = tmp_path / "icd10_dictionary.jsonl"
    write_icd10_concept_rows(output_path, rows)
    store = DictionaryStore.from_jsonl(output_path)
    assert store.exact_lookup("tiểu đường")[0].code == "E11"


def test_icd10_chapter_for_code_maps_core_chapters() -> None:
    assert icd10_chapter_for_code("A00")["chapter"] == "I"
    assert icd10_chapter_for_code("E11.9")["chapter"] == "IV"
    assert icd10_chapter_for_code("I25.1")["chapter"] == "IX"
    assert icd10_chapter_for_code("Z99")["chapter"] == "XXI"


def test_import_icd10_dictionary_cli_accepts_source_zips(tmp_path: Path) -> None:
    who_zip = tmp_path / "who.zip"
    with zipfile.ZipFile(who_zip, "w") as archive:
        archive.writestr("icd102019en.xml", WHO_CLAML_FIXTURE)
    cdc_zip = tmp_path / "cdc.zip"
    with zipfile.ZipFile(cdc_zip, "w") as archive:
        archive.writestr("icd10c-tabular.xml", CDC_TABULAR_FIXTURE)
    output_path = tmp_path / "icd10.jsonl"
    manifest_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_icd10_dictionary.py",
            "--who-claml",
            str(who_zip),
            "--cdc-xml",
            str(cdc_zip),
            "--allow-non-vietnamese-primary",
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
    assert summary["output"] == str(output_path)
    assert manifest["concepts"] == 5
    assert manifest["source_counts"]["cdc_icd10cm_2026"] == 3
    assert manifest["source_counts"]["who_icd10_2019"] == 2
    assert manifest["source_parse_counts"] == [
        {"concepts": 2, "parser": "who_claml", "path": str(who_zip)},
        {"concepts": 3, "parser": "cdc_xml", "path": str(cdc_zip)},
    ]
    assert output_path.exists()


def test_import_icd10_dictionary_cli_accepts_tt06_primary_source(tmp_path: Path) -> None:
    tt06_path = tmp_path / "tt06.csv"
    tt06_path.write_text(
        "code,official_name_vi,official_name_en,aliases,parent_code\n"
        "J18.9,\"Viêm phổi, không xác định\",\"Pneumonia, unspecified\",\"viêm phổi|VP\",J18\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "icd10.jsonl"
    manifest_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_icd10_dictionary.py",
            "--icd10-vn-tt06",
            str(tt06_path),
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
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["source_policy"] == icd10_source_policy()
    assert row["canonical_name"] == "Viêm phổi, không xác định"
    assert row["source"] == ICD10_VN_TT06_2026_SOURCE_ID
    assert manifest["source_counts"][ICD10_VN_TT06_2026_SOURCE_ID] == 1


def test_import_icd10_dictionary_cli_rejects_non_vietnamese_primary_by_default(
    tmp_path: Path,
) -> None:
    cdc_path = tmp_path / "cdc_descriptions.txt"
    cdc_path.write_text("E11 Type 2 diabetes mellitus\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_icd10_dictionary.py",
            "--cdc-descriptions",
            str(cdc_path),
            "--output",
            str(tmp_path / "icd10.jsonl"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "TT 06/2026/TT-BYT" in result.stderr


def test_import_icd10_dictionary_cli_fails_when_source_parses_zero(
    tmp_path: Path,
) -> None:
    pdf_only_zip = tmp_path / "cdc_pdf_only.zip"
    with zipfile.ZipFile(pdf_only_zip, "w") as archive:
        archive.writestr("icd10cm-CodesFile.pdf", "%PDF-1.7")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_icd10_dictionary.py",
            "--cdc-descriptions",
            str(pdf_only_zip),
            "--output",
            str(tmp_path / "icd10.jsonl"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "No ICD-10 concepts parsed" in result.stderr
