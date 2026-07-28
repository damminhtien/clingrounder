from __future__ import annotations

from medical_kg_nlp.ner.document_structure import (
    DocumentGenre,
    DocumentStructureAnalyzer,
    SectionKind,
    classify_section_heading_label,
)


def test_document_structure_preserves_crlf_offsets_and_section_scope() -> None:
    text = (
        "Tiền sử bệnh\r\n"
        "- tăng huyết áp\r\n"
        "Triệu chứng hiện tại:\r\n"
        "- đau ngực\r\n"
        "Kết quả xét nghiệm\r\n"
        "- kali: 2.4"
    )

    structure = DocumentStructureAnalyzer().analyze(text)
    pain_start = text.index("đau ngực")
    potassium_start = text.index("kali")

    assert structure.genre is DocumentGenre.CLINICAL_NOTE
    assert structure.section_at(pain_start).kind is SectionKind.SYMPTOM
    assert structure.section_at(potassium_start).kind is SectionKind.LABORATORY
    assert structure.starts_list_item(pain_start)
    assert structure.starts_list_item(potassium_start)
    assert text[slice(*structure.line_at(pain_start).span)] == "- đau ngực"


def test_document_structure_detects_numbered_medication_list() -> None:
    text = (
        "Danh sách thuốc:\n"
        "1. amlodipine 10 mg po daily\n"
        "2. aspirin 81 mg po daily"
    )

    structure = DocumentStructureAnalyzer().analyze(text)

    assert structure.genre is DocumentGenre.MEDICATION_LIST
    assert structure.starts_list_item(text.index("amlodipine"))
    assert structure.starts_list_item(text.index("aspirin"))
    assert structure.section_at(text.index("aspirin")).kind is SectionKind.MEDICATION


def test_document_structure_detects_question_answer_without_inventing_sections() -> None:
    text = (
        "Câu hỏi: Thiếu men G6PD là gì?\n"
        "Trả lời: Đây là một rối loạn di truyền.\n"
        "Hỏi: Biểu hiện thường gặp là gì?\n"
        "Đáp án: Vàng da và thiếu máu."
    )

    structure = DocumentStructureAnalyzer().analyze(text)

    assert structure.genre is DocumentGenre.QUESTION_ANSWER
    assert structure.sections == ()


def test_document_structure_requires_heading_termination() -> None:
    text = "Triệu chứng thường gặp là sốt và đau đầu trong bệnh cảnh này."

    structure = DocumentStructureAnalyzer().analyze(text)

    assert structure.genre is DocumentGenre.EDUCATIONAL
    assert structure.sections == ()


def test_document_structure_detects_roman_numbered_lab_headings() -> None:
    text = (
        "I. Kết quả xét nghiệm & Cận lâm sàng đã có\n"
        "Xét nghiệm có kết quả:\n"
        "Kết quả Cận lâm sàng\n"
        ". Xét nghiệm Máu"
    )

    structure = DocumentStructureAnalyzer().analyze(text)

    first = text.index("Cận lâm sàng")
    second = text.index("Cận lâm sàng", first + 1)
    assert structure.section_at(first).kind is SectionKind.LABORATORY
    assert structure.section_at(second).kind is SectionKind.LABORATORY
    assert structure.line_at(first).content_start == text.index("Kết quả xét nghiệm")
    assert (
        classify_section_heading_label("Cận lâm sàng")
        is SectionKind.LABORATORY
    )
