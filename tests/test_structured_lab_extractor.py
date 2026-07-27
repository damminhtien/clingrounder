from __future__ import annotations

from medical_kg_nlp.ner.contracts import RuleNerContext
from medical_kg_nlp.ner.extractors.structured_lab import (
    StructuredLabProposalExtractor,
)
from medical_kg_nlp.ner.proposal import EntityProposal
from medical_kg_nlp.schema.types import EntityType


def test_structured_lab_extracts_unknown_name_and_value_inside_lab_section() -> None:
    text = "Kết quả xét nghiệm\n- bnp 4227\n- lactate 1.1-->0.8"

    proposals = StructuredLabProposalExtractor().propose(text, RuleNerContext())

    assert _typed_mentions(text, proposals) == [
        ("bnp", EntityType.LAB_TEST),
        ("4227", EntityType.LAB_RESULT),
        ("lactate", EntityType.LAB_TEST),
        ("1.1-->0.8", EntityType.LAB_RESULT),
    ]


def test_structured_lab_supports_result_before_test_and_qualitative_result() -> None:
    text = "Kết quả xét nghiệm máu\n- 80 neutrophil\n- bình thường chem 7"

    proposals = StructuredLabProposalExtractor().propose(text, RuleNerContext())

    assert _typed_mentions(text, proposals) == [
        ("80", EntityType.LAB_RESULT),
        ("neutrophil", EntityType.LAB_TEST),
        ("bình thường", EntityType.LAB_RESULT),
        ("chem 7", EntityType.LAB_TEST),
    ]


def test_structured_lab_handles_inline_pairs_and_multiple_segments() -> None:
    text = (
        "Xét nghiệm ngoại trú được thực hiện hôm nay và canxi là 12.0; "
        "canxi ion hóa 6.8."
    )

    proposals = StructuredLabProposalExtractor().propose(text, RuleNerContext())

    assert _typed_mentions(text, proposals) == [
        ("canxi", EntityType.LAB_TEST),
        ("12.0", EntityType.LAB_RESULT),
        ("canxi ion hóa", EntityType.LAB_TEST),
        ("6.8", EntityType.LAB_RESULT),
    ]


def test_structured_lab_stops_at_next_section_and_rejects_bare_numbers() -> None:
    text = (
        "Kết quả xét nghiệm\n"
        "- kali 3.2\n"
        "Điều trị:\n"
        "- amlodipine 10 mg po daily\n"
        "Mã hồ sơ: 2026"
    )

    proposals = StructuredLabProposalExtractor().propose(text, RuleNerContext())

    assert _typed_mentions(text, proposals) == [
        ("kali", EntityType.LAB_TEST),
        ("3.2", EntityType.LAB_RESULT),
    ]


def test_structured_lab_proposals_preserve_raw_offsets() -> None:
    text = "Cận lâm sàng::\r\n- cấy máu :âm tính"

    proposals = StructuredLabProposalExtractor().propose(text, RuleNerContext())

    assert _typed_mentions(text, proposals) == [
        ("cấy máu", EntityType.LAB_TEST),
        ("âm tính", EntityType.LAB_RESULT),
    ]
    for proposal in proposals:
        proposal.validate_offsets(text)


def test_structured_lab_rejects_metadata_and_stops_at_inline_section_heading() -> None:
    text = (
        "Thời điểm khởi phát triệu chứng: 3 ngày trước\n"
        "Kết quả xét nghiệm\n"
        "- kali 3.2\n"
        "Kết quả chẩn đoán hình ảnh: gãy 3 xương sườn 9, 10 và 11\n"
        "Các thủ thuật đã thực hiện\n"
        "- Đặt 3 stent"
    )

    proposals = StructuredLabProposalExtractor().propose(text, RuleNerContext())

    assert _typed_mentions(text, proposals) == [
        ("kali", EntityType.LAB_TEST),
        ("3.2", EntityType.LAB_RESULT),
    ]


def test_structured_lab_trims_result_course_from_test_name() -> None:
    text = (
        "Kết quả xét nghiệm\n"
        "- hct trả về là 24.7\n"
        "- glucose cải thiện thành 367\n"
        "- tăng Cr theo phòng cấp cứu"
    )

    proposals = StructuredLabProposalExtractor().propose(text, RuleNerContext())

    assert _typed_mentions(text, proposals) == [
        ("hct", EntityType.LAB_TEST),
        ("24.7", EntityType.LAB_RESULT),
        ("glucose", EntityType.LAB_TEST),
        ("367", EntityType.LAB_RESULT),
        ("tăng", EntityType.LAB_RESULT),
        ("Cr", EntityType.LAB_TEST),
    ]


def test_structured_lab_does_not_retype_diagnosis_like_qualitative_prefix() -> None:
    text = "Kết quả xét nghiệm\n- suy thận cấp\n- bất thường điện giải"

    proposals = StructuredLabProposalExtractor().propose(text, RuleNerContext())

    assert proposals == ()


def _typed_mentions(
    source_text: str,
    proposals: tuple[EntityProposal, ...],
) -> list[tuple[str, EntityType]]:
    output: list[tuple[str, EntityType]] = []
    for proposal in proposals:
        entity_type = proposal.entity_type
        assert entity_type is not None
        output.append(
            (
                source_text[proposal.span[0] : proposal.span[1]],
                entity_type,
            )
        )
    return output
