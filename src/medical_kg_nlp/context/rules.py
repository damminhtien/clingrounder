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
    "có khả năng",
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
    "các sự kiện trước khi nhập viện": AssertionStatus.HISTORICAL,
    "tình trạng ngay trước khi nhập viện": AssertionStatus.HISTORICAL,
    "plan": AssertionStatus.PLANNED,
    "kế hoạch": AssertionStatus.PLANNED,
}


_SOURCE_CUES = load_default_assertion_cues()
_LEFT_CUES_BY_ASSERTION = cues_by_assertion(_SOURCE_CUES, scope="left")
_RIGHT_CUES_BY_ASSERTION = cues_by_assertion(_SOURCE_CUES, scope="right")
_BIDIRECTIONAL_CUES_BY_ASSERTION = cues_by_assertion(_SOURCE_CUES, scope="bidirectional")


def _directional_cues(
    assertion: AssertionStatus,
    *,
    direction: str,
    fallback: tuple[str, ...] = (),
) -> tuple[str, ...]:
    scoped = _LEFT_CUES_BY_ASSERTION if direction == "left" else _RIGHT_CUES_BY_ASSERTION
    values = (*scoped.get(assertion, ()), *_BIDIRECTIONAL_CUES_BY_ASSERTION.get(assertion, ()))
    if not values and direction == "left":
        values = fallback
    return tuple(dict.fromkeys(values))


POSSIBLE_LEFT_CUES = _directional_cues(
    AssertionStatus.POSSIBLE,
    direction="left",
    fallback=_FALLBACK_POSSIBLE_CUES,
)
POSSIBLE_RIGHT_CUES = _directional_cues(AssertionStatus.POSSIBLE, direction="right")
NEGATION_LEFT_CUES = _directional_cues(
    AssertionStatus.NEGATED,
    direction="left",
    fallback=_FALLBACK_NEGATION_CUES,
)
NEGATION_RIGHT_CUES = _directional_cues(AssertionStatus.NEGATED, direction="right")
HISTORICAL_LEFT_CUES = _directional_cues(
    AssertionStatus.HISTORICAL,
    direction="left",
    fallback=_FALLBACK_HISTORICAL_CUES,
)
HISTORICAL_RIGHT_CUES = _directional_cues(AssertionStatus.HISTORICAL, direction="right")
FAMILY_LEFT_CUES = _directional_cues(
    AssertionStatus.FAMILY,
    direction="left",
    fallback=_FALLBACK_FAMILY_CUES,
)
FAMILY_RIGHT_CUES = _directional_cues(AssertionStatus.FAMILY, direction="right")
PLANNED_LEFT_CUES = _directional_cues(
    AssertionStatus.PLANNED,
    direction="left",
    fallback=_FALLBACK_PLANNED_CUES,
)
PLANNED_RIGHT_CUES = _directional_cues(AssertionStatus.PLANNED, direction="right")
RESOLVED_LEFT_CUES = _directional_cues(
    AssertionStatus.RESOLVED,
    direction="left",
    fallback=_FALLBACK_RESOLVED_CUES,
)
RESOLVED_RIGHT_CUES = _directional_cues(AssertionStatus.RESOLVED, direction="right")

# Aggregate exports remain useful for dataset profiling, but classification uses
# the directional constants above.
POSSIBLE_CUES = tuple(dict.fromkeys((*POSSIBLE_LEFT_CUES, *POSSIBLE_RIGHT_CUES)))
NEGATION_CUES = tuple(dict.fromkeys((*NEGATION_LEFT_CUES, *NEGATION_RIGHT_CUES)))
HISTORICAL_CUES = tuple(dict.fromkeys((*HISTORICAL_LEFT_CUES, *HISTORICAL_RIGHT_CUES)))
FAMILY_CUES = tuple(dict.fromkeys((*FAMILY_LEFT_CUES, *FAMILY_RIGHT_CUES)))
PLANNED_CUES = tuple(dict.fromkeys((*PLANNED_LEFT_CUES, *PLANNED_RIGHT_CUES)))
RESOLVED_CUES = tuple(dict.fromkeys((*RESOLVED_LEFT_CUES, *RESOLVED_RIGHT_CUES)))

SECTION_PRIORS = {
    **_FALLBACK_SECTION_PRIORS,
    **section_priors_from_cues(_SOURCE_CUES),
}
