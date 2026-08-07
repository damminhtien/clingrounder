from pathlib import Path

import pytest

from clingrounder.dictionaries.vn_clinical_lexicon import (
    VN_CLINICAL_LEXICON_SOURCE_ID,
    build_vn_clinical_lexicon_manifest,
    parse_vn_clinical_lexicon,
)


def test_parse_vn_clinical_lexicon_writes_local_concept_rows(tmp_path: Path) -> None:
    source = tmp_path / "terms.tsv"
    source.write_text(
        "\t".join(
            [
                "code",
                "semantic_type",
                "canonical_name",
                "official_name_vi",
                "official_name_en",
                "aliases",
                "synonyms",
                "abbreviations",
                "parents",
                "notes",
            ]
        )
        + "\n"
        + "CT\tLAB_TEST\tComputed tomography\tChụp cắt lớp vi tính\tComputed tomography\tchụp CT|chụp cắt lớp\t\tCT\t\treviewed\n"
        + "PROC_SURGERY\tPROCEDURE\tSurgery\tPhẫu thuật\tSurgery\tphẫu thuật|mổ\t\tsurgery\t\tfuture phase\n",
        encoding="utf-8",
    )

    rows, warnings = parse_vn_clinical_lexicon(source)
    by_id = {row["concept_id"]: row for row in rows}
    manifest = build_vn_clinical_lexicon_manifest(rows=rows, source_inputs=[str(source)], parse_warnings=warnings)

    assert warnings == []
    assert by_id["LOCAL:TEST_CT"]["code"] == "CT"
    assert by_id["LOCAL:TEST_CT"]["code_system"] == "LOCAL"
    assert by_id["LOCAL:TEST_CT"]["semantic_type"] == "LAB_TEST"
    assert by_id["LOCAL:TEST_CT"]["source_ids"] == [VN_CLINICAL_LEXICON_SOURCE_ID]
    assert "chụp CT" in by_id["LOCAL:TEST_CT"]["aliases"]
    assert by_id["LOCAL:PROC_SURGERY"]["semantic_type"] == "PROCEDURE"
    assert manifest["concepts"] == 2
    assert manifest["by_semantic_type"] == {"LAB_TEST": 1, "PROCEDURE": 1}


@pytest.mark.private
def test_reviewed_lexicon_contains_phase1_train_split_lab_tests() -> None:
    rows, warnings = parse_vn_clinical_lexicon("data/standards/vn_clinical_lexicon/raw/reviewed_terms.tsv")
    by_id = {row["concept_id"]: row for row in rows}

    assert warnings == []
    aliases_by_id = {concept_id: {alias.casefold() for alias in row["aliases"]} for concept_id, row in by_id.items()}
    assert "nội soi" in aliases_by_id["LOCAL:TEST_ENDOSCOPY"]
    assert "cấy máu" in aliases_by_id["LOCAL:TEST_BLOOD_CULTURE"]
    assert "pcr" in aliases_by_id["LOCAL:TEST_PCR"]
    assert "chọc dò dịch não tủy" in aliases_by_id["LOCAL:TEST_LUMBAR_PUNCTURE"]
