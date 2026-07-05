from pathlib import Path

from medical_kg_nlp.dictionaries.vn_clinical_lexicon import (
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
