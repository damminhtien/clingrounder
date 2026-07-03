from __future__ import annotations
from medical_kg_nlp.context.cue_loader import (
    cues_by_assertion,
    load_default_assertion_cues,
    section_priors_from_cues,
)
from medical_kg_nlp.schema.types import AssertionStatus


_FALLBACK_POSSIBLE_CUES = (
    "không loại trừ",
    "cannot exclude",
    "rule out",
    "gợi ý",
    "nghi",
    "theo dõi",
    "có thể",
    "possible",
    "suspected",
    "suggestive of",
    "likely",
)
_FALLBACK_NEGATION_CUES = (
    "không ghi nhận",
    "không có bằng chứng",
    "không thấy",
    "âm tính với",
    "phủ nhận",
    "không có",
    "không",
    "chưa",
    "no evidence of",
    "negative for",
    "without",
    "denies",
    "denied",
    "no ",
    "not ",
)
_FALLBACK_HISTORICAL_CUES = (
    "tiền sử",
    "đã từng",
    "trước đây",
    "bệnh nền",
    "history of",
    "past medical history",
    "previous",
    "prior",
    "known case of",
)
_FALLBACK_FAMILY_CUES = (
    "family history",
    "gia đình",
    "người nhà",
    "mother",
    "father",
    "sister",
    "brother",
    "parent",
    "mẹ",
    "cha",
    "bố",
    "anh",
    "chị",
    "em",
    "ông",
    "bà",
)
_FALLBACK_PLANNED_CUES = (
    "will start",
    "planned",
    "scheduled",
    "recommend",
    "sẽ",
    "dự kiến",
    "chỉ định",
    "kế hoạch",
)
_FALLBACK_RESOLVED_CUES = ("resolved", "hết", "đã khỏi", "cải thiện")

_FALLBACK_SECTION_PRIORS = {
    "family history": AssertionStatus.FAMILY,
    "tiền sử gia đình": AssertionStatus.FAMILY,
    "past medical history": AssertionStatus.HISTORICAL,
    "tiền sử": AssertionStatus.HISTORICAL,
    "tiền sử bệnh": AssertionStatus.HISTORICAL,
    "tiền sử bệnh nội khoa": AssertionStatus.HISTORICAL,
    "các bệnh lý mạn tính": AssertionStatus.HISTORICAL,
    "thuốc trước khi nhập viện": AssertionStatus.HISTORICAL,
    "plan": AssertionStatus.PLANNED,
    "kế hoạch": AssertionStatus.PLANNED,
}


_SOURCE_CUES = load_default_assertion_cues()
_LEFT_CUES_BY_ASSERTION = cues_by_assertion(_SOURCE_CUES)

POSSIBLE_CUES = _LEFT_CUES_BY_ASSERTION.get(AssertionStatus.POSSIBLE, _FALLBACK_POSSIBLE_CUES)
NEGATION_CUES = _LEFT_CUES_BY_ASSERTION.get(AssertionStatus.NEGATED, _FALLBACK_NEGATION_CUES)
HISTORICAL_CUES = _LEFT_CUES_BY_ASSERTION.get(AssertionStatus.HISTORICAL, _FALLBACK_HISTORICAL_CUES)
FAMILY_CUES = _LEFT_CUES_BY_ASSERTION.get(AssertionStatus.FAMILY, _FALLBACK_FAMILY_CUES)
PLANNED_CUES = _LEFT_CUES_BY_ASSERTION.get(AssertionStatus.PLANNED, _FALLBACK_PLANNED_CUES)
RESOLVED_CUES = _LEFT_CUES_BY_ASSERTION.get(AssertionStatus.RESOLVED, _FALLBACK_RESOLVED_CUES)

SECTION_PRIORS = {
    **_FALLBACK_SECTION_PRIORS,
    **section_priors_from_cues(_SOURCE_CUES),
}
