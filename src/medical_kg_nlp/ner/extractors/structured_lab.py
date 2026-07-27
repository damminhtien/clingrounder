"""Structure-first proposals for laboratory tables with incomplete terminology coverage."""

from __future__ import annotations

import re
from dataclasses import dataclass

from medical_kg_nlp.ner.contracts import RuleNerContext
from medical_kg_nlp.ner.proposal import EntityProposal
from medical_kg_nlp.schema.types import AssertionStatus, EntityType
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["StructuredLabProposalExtractor"]


_LINE_RE = re.compile(r"[^\r\n]+")
_ITEM_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+\s*|\d{1,3}[.)]\s*)", re.UNICODE)
_LAB_HEADING_RE = re.compile(
    r"^(?:"
    r"kết\s+quả\s+(?:xét\s*nghiệm(?:\s+máu)?|xétí\s*nghiệm|phòng\s+thí\s+nghiệm)|"
    r"xét\s*nghiệm(?:\s+máu)?|"
    r"cận\s+lâm\s+sàng|"
    r"laboratory(?:\s+results?)?|"
    r"lab(?:s|\s+results?)?"
    r")\s*:*$",
    re.IGNORECASE | re.UNICODE,
)
_LINE_LAB_CONTEXT_RE = re.compile(
    r"^(?:xét\s*nghiệm|kết\s+quả\s+(?:xét\s*nghiệm|phòng\s+thí\s+nghiệm))\b",
    re.IGNORECASE | re.UNICODE,
)
_INLINE_LAB_HEADING_RE = re.compile(
    r"^(?:kết\s+quả\s+(?:xét\s*nghiệm|xétí\s*nghiệm|phòng\s+thí\s+nghiệm)|"
    r"cận\s+lâm\s+sàng)\s*:+\s*",
    re.IGNORECASE | re.UNICODE,
)
_NON_LAB_SECTION_PREFIX_RE = re.compile(
    r"^(?:"
    r"kết\s+quả\s+chẩn\s+đoán\s+hình\s+ảnh|"
    r"chẩn\s+đoán\s+hình\s+ảnh|"
    r"các\s+kết\s+quả\s+chẩn\s+đoán\s+khác|"
    r"hình\s+ảnh|"
    r"triệu\s+chứng|"
    r"tiền\s+sử|"
    r"bệnh\s+sử|"
    r"lý\s+do\s+(?:nhập|vào)\s+viện|"
    r"khám(?:\s+lâm\s+sàng)?|"
    r"chẩn\s+đoán|"
    r"điều\s+trị|"
    r"thuốc|"
    r"(?:các\s+)?thủ\s+thuật(?:\s+đã)?\s+thực\s+hiện|"
    r"thủ\s+thuật|"
    r"kế\s+hoạch|"
    r"đánh\s+giá\s+tại\s+bệnh\s+viện|"
    r"kết\s+quả\s+khám\s+thực\s+thể|"
    r"dấu\s+hiệu\s+lâm\s+sàng"
    r")(?=\s*:|$)",
    re.IGNORECASE | re.UNICODE,
)
_EXPLICIT_PAIR_CUE_RE = re.compile(r"(?::|=|→|--?>|\blà\b)", re.IGNORECASE)
_UNIT = (
    r"(?:%|mmhg|mmol/l|mg/dl|g/dl|g/l|ng/ml|meq/l|iu/l|u/l|"
    r"[uµμ]mol/l|g/l|10\^?\d+/l|°?\s*c)"
)
_NUMERIC_RESULT_RE = re.compile(
    r"(?<![\w])"
    r"(?P<value>[<>]?\s*\d+(?:[.,]\d+)*"
    r"(?:\s*(?:--?>|→)\s*[<>]?\s*\d+(?:[.,]\d+)*)?"
    r"(?:\s*[x*]\s*\d+(?:[.,]\d+)?){0,2}"
    rf"(?:\s*{_UNIT})?)"
    r"(?![\w])",
    re.IGNORECASE | re.UNICODE,
)
_QUALITATIVE_RESULT_RE = re.compile(
    r"(?<!\w)(?P<value>"
    r"không\s+ghi\s+nhận\s+gì\s+bất\s+thường|"
    r"không\s+có\s+gì\s+(?:đáng\s+chú\s+ý|bất\s+thường)|"
    r"không\s+(?:thấy|phát\s+hiện)\s+[^,;]{1,48}|"
    r"dưới\s+ngưỡng\s+điều\s+trị(?:\s+\d+(?:[.,]\d+)*)?|"
    r"tăng\s+nhẹ\s+lên\s+\d+(?:[.,]\d+)*|"
    r"xu\s+hướng\s+(?:tăng|giảm)|"
    r"đang\s+chờ(?:\s+kết\s+quả)?|"
    r"dương\s+tính|âm\s+tính|bình\s+thường|bất\s+thường|"
    r"tăng\s+cao|giảm\s+thấp|tăng|giảm|cao|thấp"
    r")(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
_LEADING_NAME_CUE_RE = re.compile(r"^(?:đo|giá\s+trị|chỉ\s+số)\s+", re.IGNORECASE)
_TRAILING_PAIR_CUE_RE = re.compile(
    r"\s*(?::|=|→|--?>|\blà\b)\s*$",
    re.IGNORECASE | re.UNICODE,
)
_NAME_TRAILING_CONTEXT_RE = re.compile(
    r"\s+(?:"
    r"của\s+(?:bệnh\s+nhân|anh\s+ấy|cô\s+ấy)|"
    r"đã\s+có|"
    r"trả\s+về|"
    r"vào\s+ngày(?:\s+\S+){0,3}|"
    r"lặp\s+lại(?:\s+tại)?|"
    r"vẫn(?:\s+đang|\s+giảm|\s+tăng)?|"
    r"cải\s+thiện(?:\s+thành)?|"
    r"bắt\s+đầu|"
    r"cho\s+thấy|"
    r"(?:tại|theo)\s+(?:phòng|khoa|cơ\s+sở)"
    r")\b.*$",
    re.IGNORECASE | re.UNICODE,
)
_BANNED_NAME_RE = re.compile(
    r"^(?:"
    r"ngày|tháng|năm|tuổi|mã|liều|thuốc|bệnh\s+nhân|điều\s+trị|"
    r"uống|tiêm|truyền|chẩn\s+đoán|thủ\s+thuật|số\s+lượng|"
    r"thời\s+(?:điểm|gian)|tần\s+suất|mức\s+độ|vị\s+trí|"
    r"triệu\s+chứng|sốt|mạch|dấu\s+hiệu\s+sinh\s+tồn|"
    r"so\s+với|nhưng|sau|trước|được\s+dùng"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class _LabPair:
    test_span: tuple[int, int]
    result_span: tuple[int, int]
    rule_id: str


@dataclass(frozen=True, slots=True)
class StructuredLabProposalExtractor:
    """Infer test/result pairs only from explicit or section-scoped tabular structure.

    INVARIANT: a bare number never becomes a result outside an identified lab section unless
    the same segment contains an explicit ``:``, ``=``, arrow, or ``là`` relation.
    """

    def propose(
        self,
        source_text: str,
        context: RuleNerContext,
    ) -> tuple[EntityProposal, ...]:
        del context
        proposals: set[EntityProposal] = set()
        in_lab_section = False
        for line_match in _LINE_RE.finditer(source_text):
            line_span = _trim_span(source_text, line_match.span())
            if line_span[0] >= line_span[1]:
                continue
            logical_span = _strip_item_prefix(source_text, line_span)
            line = source_text[logical_span[0] : logical_span[1]]
            normalized_line = normalize_for_match(line.strip())
            inline_heading = _INLINE_LAB_HEADING_RE.match(line)
            if _LAB_HEADING_RE.fullmatch(normalized_line):
                in_lab_section = True
                continue
            if _NON_LAB_SECTION_PREFIX_RE.match(line):
                # A section header may share a physical line with prose. Treat the complete
                # line as owned by that section so lab state cannot leak into imaging or
                # procedure numbers.
                in_lab_section = False
                continue

            content_start = logical_span[0]
            section_context = in_lab_section or bool(_LINE_LAB_CONTEXT_RE.match(line))
            if inline_heading is not None:
                content_start = logical_span[0] + inline_heading.end()
                section_context = True
                in_lab_section = True
            content_span = _strip_item_prefix(
                source_text,
                _trim_span(source_text, (content_start, line_span[1])),
            )
            for segment_span in _segment_spans(source_text, content_span):
                pair = _parse_pair(
                    source_text,
                    segment_span,
                    section_context=section_context,
                )
                if pair is None:
                    continue
                confidence = 0.86 if section_context else 0.82
                features = (
                    ("default_assertion", AssertionStatus.PRESENT.value),
                    ("rule_id", pair.rule_id),
                    ("section_context", str(section_context).lower()),
                )
                proposals.add(
                    EntityProposal(
                        span=pair.test_span,
                        candidate_types=(EntityType.LAB_TEST,),
                        source="structured_lab",
                        score=confidence,
                        evidence_ids=(f"structured_lab:{pair.rule_id}:test",),
                        features=tuple(sorted(features)),
                    )
                )
                proposals.add(
                    EntityProposal(
                        span=pair.result_span,
                        candidate_types=(EntityType.LAB_RESULT,),
                        source="structured_lab",
                        score=confidence,
                        evidence_ids=(f"structured_lab:{pair.rule_id}:result",),
                        features=tuple(sorted(features)),
                    )
                )
        return tuple(sorted(proposals, key=_proposal_order))


def _parse_pair(
    source_text: str,
    segment_span: tuple[int, int],
    *,
    section_context: bool,
) -> _LabPair | None:
    start, end = _strip_item_prefix(source_text, segment_span)
    if start >= end:
        return None
    segment = source_text[start:end]

    qualitative_prefix = _QUALITATIVE_RESULT_RE.match(segment)
    if (
        qualitative_prefix is not None
        and section_context
        and _allows_qualitative_result_before_test(
            source_text,
            start,
            end,
            qualitative_prefix,
        )
    ):
        result_span = _trim_span(
            source_text,
            _absolute_span(start, qualitative_prefix.span("value")),
        )
        test_span = _clean_test_name_span(
            source_text,
            (result_span[1], end),
        )
        if _plausible_test_name(source_text, test_span):
            return _LabPair(test_span, result_span, "result_before_test")

    numeric_prefix = _NUMERIC_RESULT_RE.match(segment)
    if numeric_prefix is not None and section_context:
        result_span = _trim_span(
            source_text,
            _absolute_span(start, numeric_prefix.span("value")),
        )
        test_span = _clean_test_name_span(source_text, (result_span[1], end))
        if _plausible_test_name(source_text, test_span):
            return _LabPair(test_span, result_span, "result_before_test")

    result_matches = [
        *(_QUALITATIVE_RESULT_RE.finditer(segment)),
        *(_NUMERIC_RESULT_RE.finditer(segment)),
    ]
    result_matches.sort(key=lambda match: (match.start("value"), -match.end("value")))
    for result_match in result_matches:
        raw_test_span = (start, start + result_match.start("value"))
        test_span = _clean_test_name_span(source_text, raw_test_span)
        if not _plausible_test_name(source_text, test_span):
            continue
        gap = source_text[test_span[1] : start + result_match.start("value")]
        if not section_context and _EXPLICIT_PAIR_CUE_RE.search(gap) is None:
            continue
        result_span = _trim_span(
            source_text,
            _absolute_span(start, result_match.span("value")),
        )
        return _LabPair(test_span, result_span, "test_before_result")
    return None


def _allows_qualitative_result_before_test(
    source_text: str,
    segment_start: int,
    segment_end: int,
    result_match: re.Match[str],
) -> bool:
    """Reject diagnosis-like ``bất thường X`` while retaining lab table inversions."""

    value = normalize_for_match(result_match.group("value"))
    if value.startswith(("bình thường", "âm tính", "dương tính")):
        return True
    test_span = _clean_test_name_span(
        source_text,
        (segment_start + result_match.end("value"), segment_end),
    )
    test_name = source_text[test_span[0] : test_span[1]]
    # Compact uppercase abbreviations are common in result-first lab rows (for example
    # ``tăng Cr``). Generic lower-case nouns are more often diagnoses or prose.
    return bool(
        re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,7}", test_name)
        and any(character.isupper() for character in test_name)
    )


def _clean_test_name_span(
    source_text: str,
    span: tuple[int, int],
) -> tuple[int, int]:
    start, end = _trim_span(source_text, span)
    if start >= end:
        return start, end
    text = source_text[start:end]

    trailing_cue = _TRAILING_PAIR_CUE_RE.search(text)
    if trailing_cue is not None:
        end = start + trailing_cue.start()
        start, end = _trim_span(source_text, (start, end))
        text = source_text[start:end]

    colon = max(text.rfind(":"), text.rfind("="), text.rfind("→"))
    if colon >= 0:
        start += colon + 1
        start, end = _trim_span(source_text, (start, end))
        text = source_text[start:end]

    conjunctions = list(
        re.finditer(r"\b(?:và|and)\b", text, flags=re.IGNORECASE | re.UNICODE)
    )
    if conjunctions and re.search(
        r"\b(?:thực\s+hiện|hôm\s+nay|kết\s+quả)\b",
        text[: conjunctions[-1].start()],
        flags=re.IGNORECASE | re.UNICODE,
    ):
        start += conjunctions[-1].end()
        start, end = _trim_span(source_text, (start, end))
        text = source_text[start:end]

    leading = _LEADING_NAME_CUE_RE.match(text)
    if leading is not None:
        start += leading.end()
        start, end = _trim_span(source_text, (start, end))
        text = source_text[start:end]

    trailing = _NAME_TRAILING_CONTEXT_RE.search(text)
    if trailing is not None:
        end = start + trailing.start()
    return _trim_span(source_text, (start, end))


def _plausible_test_name(
    source_text: str,
    span: tuple[int, int],
) -> bool:
    start, end = span
    if start >= end:
        return False
    mention = source_text[start:end]
    normalized = normalize_for_match(mention)
    tokens = normalized.split()
    return (
        2 <= len(mention) <= 64
        and 1 <= len(tokens) <= 9
        and any(character.isalpha() for character in mention)
        and _BANNED_NAME_RE.match(normalized) is None
        and normalized
        not in {
            "kết quả",
            "xét nghiệm",
            "kết quả xét nghiệm",
            "cận lâm sàng",
            "bình thường",
            "bất thường",
        }
    )


def _segment_spans(
    source_text: str,
    span: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    start, end = span
    boundaries = [start]
    for index in range(start, end):
        character = source_text[index]
        if character == ";" or (
            character == ","
            and not (
                index > start
                and index + 1 < end
                and source_text[index - 1].isdigit()
                and source_text[index + 1].isdigit()
            )
        ):
            boundaries.extend((index, index + 1))
    boundaries.append(end)
    segments = [
        _trim_span(source_text, (boundaries[index], boundaries[index + 1]))
        for index in range(0, len(boundaries) - 1, 2)
    ]
    return tuple(segment for segment in segments if segment[0] < segment[1])


def _strip_item_prefix(
    source_text: str,
    span: tuple[int, int],
) -> tuple[int, int]:
    start, end = _trim_span(source_text, span)
    match = _ITEM_PREFIX_RE.match(source_text[start:end])
    if match is not None:
        start += match.end()
    return _trim_span(source_text, (start, end))


def _trim_span(source_text: str, span: tuple[int, int]) -> tuple[int, int]:
    start, end = span
    while start < end and source_text[start].isspace():
        start += 1
    while end > start and source_text[end - 1].isspace():
        end -= 1
    return start, end


def _absolute_span(base: int, relative: tuple[int, int]) -> tuple[int, int]:
    return base + relative[0], base + relative[1]


def _proposal_order(proposal: EntityProposal) -> tuple[int, int, str]:
    return proposal.span[0], proposal.span[1], proposal.candidate_types[0].value
