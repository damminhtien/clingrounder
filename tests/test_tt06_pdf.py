from pathlib import Path

from medical_kg_nlp.dictionaries.tt06_pdf import extract_tt06_rows_from_tsv


def test_extract_tt06_rows_from_tsv_uses_table_columns_and_excludes_guidance(tmp_path: Path) -> None:
    tsv_path = tmp_path / "tt06.tsv"
    tsv_path.write_text(
        "\n".join(
            [
                "level\tpage_num\tpar_num\tblock_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
                _word("1", 19.25, 114.61),
                _word("A00.0", 473.0, 114.61),
                _word("A000", 490.1, 114.61),
                _word("Cholera", 509.9, 114.61),
                _word("due", 525.0, 114.61),
                _word("to", 532.0, 114.61),
                _word("Bệnh", 591.2, 114.61),
                _word("tả", 601.1, 114.61),
                _word("do", 606.0, 114.61),
                _word("vi", 612.0, 114.61),
                _word("khuẩn", 616.0, 114.61),
                _word("Vibrio", 628.0, 114.61),
                _word("Bệnh", 641.9, 114.61),
                _word("tả", 652.0, 114.61),
                _word("Vibrio", 509.9, 120.61),
                _word("cholerae", 522.0, 120.61),
                _word("cholerae", 591.2, 120.61),
                _word("01,", 608.0, 120.61),
                _word("típ", 614.0, 120.61),
                _word("sinh", 620.0, 120.61),
                _word("học", 629.0, 120.61),
                _word("cholerae", 591.2, 126.61),
                _word("2", 19.25, 132.61),
                _word("A00.9", 473.0, 132.61),
                _word("Cholera,", 509.9, 132.61),
                _word("unspecified", 526.0, 132.61),
                _word("Bệnh", 591.2, 132.61),
                _word("tả,", 601.1, 132.61),
                _word("không", 607.0, 132.61),
                _word("xác", 619.0, 132.61),
                _word("định", 626.0, 132.61),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = extract_tt06_rows_from_tsv(tsv_path)

    assert [row.code for row in rows] == ["A00.0", "A00.9"]
    assert rows[0].official_name_en == "Cholera due to Vibrio cholerae"
    assert rows[0].official_name_vi == "Bệnh tả do vi khuẩn Vibrio cholerae 01, típ sinh học cholerae"
    assert "Bệnh tả cổ" not in rows[0].official_name_vi
    assert rows[1].official_name_vi == "Bệnh tả, không xác định"


def test_extract_tt06_rows_accepts_code_when_merged_stt_cell_is_blank(tmp_path: Path) -> None:
    tsv_path = tmp_path / "tt06.tsv"
    tsv_path.write_text(
        "\n".join(
            [
                "level\tpage_num\tpar_num\tblock_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
                _word("4867", 19.25, 114.61),
                _word("K74.5", 473.0, 114.61),
                _word("Biliary", 509.9, 114.61),
                _word("cirrhosis", 520.0, 114.61),
                _word("Xơ", 591.2, 114.61),
                _word("gan", 600.0, 114.61),
                # K74.6 belongs to the same merged STT cell, so no word appears at x=19.25.
                _word("K74.6", 473.0, 120.61),
                _word("Other", 509.9, 120.61),
                _word("cirrhosis", 520.0, 120.61),
                _word("Xơ", 591.2, 120.61),
                _word("gan", 600.0, 120.61),
                _word("khác", 608.0, 120.61),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = extract_tt06_rows_from_tsv(tsv_path)

    assert [row.code for row in rows] == ["K74.5", "K74.6"]
    assert rows[0].stt == "4867"
    assert rows[1].stt == ""
    assert rows[1].official_name_vi == "Xơ gan khác"


def test_extract_tt06_rows_does_not_treat_raw_double_quote_as_tsv_quoting(
    tmp_path: Path,
) -> None:
    tsv_path = tmp_path / "tt06.tsv"
    tsv_path.write_text(
        "\n".join(
            [
                "level\tpage_num\tpar_num\tblock_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
                _word("1", 19.25, 114.61),
                _word("A00", 473.0, 114.61),
                _word('Cholera"', 509.9, 114.61),
                _word("Bệnh", 591.2, 114.61),
                _word("tả", 601.1, 114.61),
                _word("2", 19.25, 120.61),
                _word("A01", 473.0, 120.61),
                _word("Typhoid", 509.9, 120.61),
                _word("Thương", 591.2, 120.61),
                _word("hàn", 605.0, 120.61),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = extract_tt06_rows_from_tsv(tsv_path)

    assert [row.code for row in rows] == ["A00", "A01"]


def _word(text: str, left: float, top: float) -> str:
    return f"5\t1\t0\t0\t0\t0\t{left}\t{top}\t1\t1\t100\t{text}"
