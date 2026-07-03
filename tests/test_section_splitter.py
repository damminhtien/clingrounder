from medical_kg_nlp.preprocessing.section_splitter import split_sections


def test_split_sections_recognizes_numbered_headings_and_subheadings() -> None:
    text = (
        "1.  Tiền sử bệnh\n"
        "    Thuốc trước khi nhập viện\n"
        "    - metoprolol 25mg po bid\n"
        "    Các bệnh lý mạn tính: đái tháo đường\n"
        "2. Bệnh sử hiện tại\n"
        "    Triệu chứng hiện tại\n"
        "    - đánh trống ngực\n"
        "3. Đánh giá tại bệnh viện\n"
        "    Kết quả xét nghiệm: troponin 0.2 ng/mL\n"
    )

    sections = split_sections(text)
    titles = [section.title for section in sections]

    assert "Tiền sử bệnh" in titles
    assert "Thuốc trước khi nhập viện" in titles
    assert "Các bệnh lý mạn tính" in titles
    assert "Bệnh sử hiện tại" in titles
    assert "Triệu chứng hiện tại" in titles
    assert "Đánh giá tại bệnh viện" in titles
    assert "Kết quả xét nghiệm" in titles
    medication_section = next(section for section in sections if section.title == "Thuốc trước khi nhập viện")
    assert "metoprolol" in medication_section.text
    assert text[medication_section.span[0] : medication_section.span[1]] == medication_section.text


def test_split_sections_recognizes_inline_subheading_content() -> None:
    text = "Thuốc trước khi nhập viện: aspirin 325mg hằng ngày\n2. Bệnh sử hiện tại\nKhông đau ngực."

    sections = split_sections(text)

    assert sections[0].title == "Thuốc trước khi nhập viện"
    assert sections[0].text.startswith("aspirin 325mg")
    assert sections[1].title == "Bệnh sử hiện tại"


def test_split_sections_recognizes_preadmission_status_heading() -> None:
    text = (
        "1. Tiền sử bệnh hiện tại\n"
        "Tình trạng ngay trước khi nhập viện: Tiếp tục cảm thấy đánh trống ngực.\n"
        "2. Đánh giá tại bệnh viện\n"
        "Không ghi nhận đau ngực.\n"
    )

    sections = split_sections(text)
    titles = [section.title for section in sections]

    assert "Tình trạng ngay trước khi nhập viện" in titles
    status_section = next(section for section in sections if section.title == "Tình trạng ngay trước khi nhập viện")
    assert status_section.text.startswith("Tiếp tục cảm thấy đánh trống ngực")
    assert text[status_section.span[0] : status_section.span[1]] == status_section.text
