from scripts.merge_standard_dictionaries import merge_standard_rows


def test_merge_standard_rows_enriches_seed_and_adds_only_input_matched_new_rows() -> None:
    base = [
        {
            "concept_id": "ICD10:E11",
            "code": "E11",
            "code_system": "ICD-10",
            "canonical_name": "Type 2 diabetes mellitus",
            "semantic_type": "DISEASE",
            "aliases": ["tiểu đường"],
            "source_ids": ["seed"],
        }
    ]
    standards = [
        {
            "concept_id": "ICD10:E11",
            "code": "E11",
            "code_system": "ICD-10",
            "canonical_name": "Bệnh đái tháo đường típ 2",
            "official_name_vi": "Bệnh đái tháo đường típ 2",
            "parent_code": "E10-E14",
            "icd10_chapter": "IV",
            "icd10_chapter_range": "E00-E90",
            "icd10_chapter_name_en": "Endocrine, nutritional and metabolic diseases",
            "icd10_block": "E10-E14",
            "semantic_type": "DISEASE",
            "source": "icd10_vn_tt06_2026",
        },
        {
            "concept_id": "ICD10:J84.9",
            "code": "J84.9",
            "code_system": "ICD-10",
            "canonical_name": "Bệnh phổi mô kẽ, không xác định",
            "official_name_vi": "Bệnh phổi mô kẽ, không xác định",
            "semantic_type": "DISEASE",
            "source": "icd10_vn_tt06_2026",
        },
        {
            "concept_id": "ICD10:A00.0",
            "code": "A00.0",
            "code_system": "ICD-10",
            "canonical_name": "Bệnh tả do vi khuẩn Vibrio cholerae",
            "official_name_vi": "Bệnh tả do vi khuẩn Vibrio cholerae",
            "semantic_type": "DISEASE",
            "source": "icd10_vn_tt06_2026",
        },
        {
            "concept_id": "ICD10:R06.0",
            "code": "R06.0",
            "code_system": "ICD-10",
            "canonical_name": "Khó thở",
            "official_name_vi": "Khó thở",
            "semantic_type": "DISEASE",
            "source": "icd10_vn_tt06_2026",
        },
    ]

    rows, summary = merge_standard_rows(
        base,
        standards,
        normalized_input_text=" bệnh nhân có bệnh phổi mô kẽ ",
    )
    by_id = {row["concept_id"]: row for row in rows}

    assert summary["enriched_rows"] == 1
    assert summary["added_rows"] == 1
    assert summary["skipped_rows"] == 2
    assert by_id["ICD10:E11"]["canonical_name"] == "Type 2 diabetes mellitus"
    assert by_id["ICD10:E11"]["icd10_chapter"] == "IV"
    assert by_id["ICD10:E11"]["icd10_block"] == "E10-E14"
    assert "Bệnh đái tháo đường típ 2" in by_id["ICD10:E11"]["aliases"]
    assert by_id["ICD10:J84.9"]["code"] == "J84.9"
    assert "ICD10:A00.0" not in by_id
    assert "ICD10:R06.0" not in by_id


def test_merge_standard_rows_adds_new_rxnorm_terms_only_in_drug_context() -> None:
    standards = [
        {
            "concept_id": "RXNORM:6470",
            "code": "6470",
            "code_system": "RxNorm",
            "canonical_name": "lorazepam",
            "semantic_type": "DRUG",
            "source": "rxnorm_prescribable_2026_06_01",
        },
        {
            "concept_id": "RXNORM:114202",
            "code": "114202",
            "code_system": "RxNorm",
            "canonical_name": "lactate",
            "semantic_type": "DRUG",
            "source": "rxnorm_prescribable_2026_06_01",
        },
    ]

    rows, summary = merge_standard_rows(
        [],
        standards,
        normalized_input_text=" bệnh nhân được dùng lorazepam 1 mg . lactate tăng ",
    )
    by_id = {row["concept_id"]: row for row in rows}

    assert summary["added_rows"] == 1
    assert "RXNORM:6470" in by_id
    assert "RXNORM:114202" not in by_id


def test_merge_standard_rows_backfills_existing_rows_by_code_without_duplicates() -> None:
    base = [
        {
            "concept_id": "LEGACY:E11",
            "code": "E11",
            "code_system": "ICD-10",
            "canonical_name": "Type 2 diabetes mellitus",
            "semantic_type": "DISEASE",
            "source_ids": ["seed"],
        },
        {
            "concept_id": "RXNORM:6809",
            "code": "6809",
            "code_system": "RxNorm",
            "canonical_name": "metformin",
            "semantic_type": "DRUG",
            "source_ids": ["seed"],
        },
    ]
    standards = [
        {
            "concept_id": "ICD10:E11",
            "code": "E11",
            "code_system": "ICD-10",
            "canonical_name": "Đái tháo đường típ 2",
            "official_name_vi": "Đái tháo đường típ 2",
            "semantic_type": "DISEASE",
            "parent_code": "E10-E14",
            "icd10_chapter": "IV",
            "icd10_chapter_range": "E00-E90",
            "icd10_block": "E10-E14",
            "source": "icd10_vn_tt06_2026",
        },
        {
            "concept_id": "RXNORM:6809",
            "code": "6809",
            "code_system": "RxNorm",
            "canonical_name": "metformin",
            "semantic_type": "DRUG",
            "ingredient": "metformin",
            "dose_form": "Oral Tablet",
            "strength": "RXN_STRENGTH=500 MG",
            "rxnorm_status": "active",
            "rxnorm_ttys": ["IN", "SCD"],
            "source": "rxnorm_prescribable_2026_06_01",
        },
    ]

    rows, summary = merge_standard_rows(base, standards)
    by_id = {row["concept_id"]: row for row in rows}

    assert summary["output_rows"] == 2
    assert summary["enriched_rows"] == 2
    assert summary["code_matched_enriched_rows"] == 1
    assert "ICD10:E11" not in by_id
    assert by_id["LEGACY:E11"]["official_name_vi"] == "Đái tháo đường típ 2"
    assert by_id["LEGACY:E11"]["icd10_chapter"] == "IV"
    assert by_id["LEGACY:E11"]["icd10_block"] == "E10-E14"
    assert by_id["RXNORM:6809"]["ingredient"] == "metformin"
    assert by_id["RXNORM:6809"]["dose_form"] == "Oral Tablet"
    assert by_id["RXNORM:6809"]["rxnorm_status"] == "active"
    assert by_id["RXNORM:6809"]["rxnorm_ttys"] == ["IN", "SCD"]


def test_merge_standard_rows_derives_icd_hierarchy_for_seed_only_codes() -> None:
    base = [
        {
            "concept_id": "ICD10:A04.72",
            "code": "A04.72",
            "code_system": "ICD-10",
            "canonical_name": "Enterocolitis due to Clostridium difficile",
            "semantic_type": "DISEASE",
            "parent_code": "A04.7",
            "source_ids": ["cdc_icd10cm_2026"],
        }
    ]

    rows, summary = merge_standard_rows(base, [])
    row = rows[0]

    assert summary["output_rows"] == 1
    assert row["icd10_chapter"] == "I"
    assert row["icd10_chapter_range"] == "A00-B99"
    assert row["icd10_block"] == "A04.7"


def test_merge_standard_rows_can_gate_new_rows_by_semantic_type() -> None:
    standards = [
        {
            "concept_id": "LOCAL:SYMPTOM_HEMOPTYSIS",
            "code": "SYMPTOM_HEMOPTYSIS",
            "code_system": "LOCAL",
            "canonical_name": "Hemoptysis",
            "official_name_vi": "Ho ra máu",
            "semantic_type": "SYMPTOM",
            "aliases": ["ho ra máu"],
            "source": "vn_clinical_lexicon_reviewed_2026_07_05",
        },
        {
            "concept_id": "LOCAL:PROC_PROCEDURE",
            "code": "PROC_PROCEDURE",
            "code_system": "LOCAL",
            "canonical_name": "Procedure",
            "official_name_vi": "Thủ thuật",
            "semantic_type": "PROCEDURE",
            "aliases": ["thủ thuật"],
            "source": "vn_clinical_lexicon_reviewed_2026_07_05",
        },
    ]

    rows, summary = merge_standard_rows(
        [],
        standards,
        normalized_input_text=" bệnh nhân ho ra máu và đã làm thủ thuật ",
        allowed_new_semantic_types={"SYMPTOM", "LAB_TEST"},
    )
    by_id = {row["concept_id"]: row for row in rows}

    assert summary["added_rows"] == 1
    assert summary["skipped_rows"] == 1
    assert summary["allowed_new_semantic_types"] == ["LAB_TEST", "SYMPTOM"]
    assert "LOCAL:SYMPTOM_HEMOPTYSIS" in by_id
    assert "LOCAL:PROC_PROCEDURE" not in by_id
