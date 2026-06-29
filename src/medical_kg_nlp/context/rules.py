from __future__ import annotations
from medical_kg_nlp.schema.types import AssertionStatus


POSSIBLE_CUES = (
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
NEGATION_CUES = (
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
HISTORICAL_CUES = (
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
FAMILY_CUES = (
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
PLANNED_CUES = (
    "will start",
    "planned",
    "scheduled",
    "recommend",
    "sẽ",
    "dự kiến",
    "chỉ định",
    "kế hoạch",
)
RESOLVED_CUES = ("resolved", "hết", "đã khỏi", "cải thiện")

SECTION_PRIORS = {
    "family history": AssertionStatus.FAMILY,
    "tiền sử gia đình": AssertionStatus.FAMILY,
    "past medical history": AssertionStatus.HISTORICAL,
    "tiền sử": AssertionStatus.HISTORICAL,
    "plan": AssertionStatus.PLANNED,
    "kế hoạch": AssertionStatus.PLANNED,
}

