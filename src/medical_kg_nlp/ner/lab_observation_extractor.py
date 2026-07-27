from __future__ import annotations

import re
from collections.abc import Iterable

from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match


_MAX_RESULT_DISTANCE = 100
_CLAUSE_END_RE = re.compile(r"[\n\r;!?]|(?<!\d)\.|\.(?!\d)")
_NUMERIC_RESULT_RE = re.compile(
    r"^\s*"
    r"(?:\([^()\n\r]{1,40}\)\s*)?"
    r"(?:(?::|=|→|->)\s*)?"
    r"(?:là\s+)?"
    r"(?P<value>[<>]?\s*\d+(?:[.,]\d+)?(?:/\d+)?(?:-\d+(?:[.,]\d+)?)?"
    r"(?:\s*%|\s*mmhg|\s*mmol/l|\s*mg/dl|\s*g/dl|\s*meq/l|\s*iu/l|\s*u/l|\s*ng/ml|\s*°?\s*c)?)"
    r"(?![\w/.,%°])",
    flags=re.IGNORECASE | re.UNICODE,
)
_QUALITATIVE_RESULT_RE = re.compile(
    r"(?<!\w)(?:"
    r"không\s+ghi\s+nhận\s+gì\s+bất\s+thường|"
    r"không\s+có\s+gì\s+(?:đáng\s+chú\s+ý|bất\s+thường)|"
    r"dương\s+tính|âm\s+tính|bình\s+thường|bất\s+thường|tăng\s+cao|tăng|giảm|cao|thấp"
    r")(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_LEFT_QUALITATIVE_RESULT_RE = re.compile(
    r"(?<!\w)(?P<value>"
    r"dương\s+tính|âm\s+tính|bình\s+thường|bất\s+thường|"
    r"đang\s+chờ(?:\s+kết\s+quả)?|"
    r"tăng\s+cao|tăng|giảm|cao|thấp"
    r")\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)


class LabObservationExtractor:
    def extract(
        self,
        text: str,
        lab_tests: Iterable[EntityAnnotation],
        *,
        occupied: Iterable[tuple[int, int]] = (),
    ) -> list[EntityAnnotation]:
        anchors = sorted(
            (entity for entity in lab_tests if entity.type == EntityType.LAB_TEST),
            key=lambda entity: entity.span,
        )
        occupied_spans = list(occupied)
        results: list[EntityAnnotation] = []
        for index, anchor in enumerate(anchors):
            previous_anchor_end = anchors[index - 1].span[1] if index > 0 else 0
            next_anchor_start = anchors[index + 1].span[0] if index + 1 < len(anchors) else len(text)
            scope_end = self._scope_end(text, anchor.span[1], next_anchor_start)
            spans = self._left_candidate_spans(text, anchor.span[0], previous_anchor_end)
            if scope_end > anchor.span[1]:
                scope = text[anchor.span[1] : scope_end]
                spans.extend(self._candidate_spans(scope, base_offset=anchor.span[1]))
            for span in spans:
                if _overlaps(span, occupied_spans):
                    continue
                occupied_spans.append(span)
                mention = text[span[0] : span[1]]
                results.append(
                    EntityAnnotation(
                        id="",
                        span=span,
                        text=mention,
                        normalized_text=normalize_for_match(mention),
                        type=EntityType.LAB_RESULT,
                        assertion=AssertionStatus.PRESENT,
                        code_system=CodeSystem.NONE,
                        confidence=0.82,
                    )
                )
        return sorted(results, key=lambda entity: entity.span)

    @staticmethod
    def _scope_end(text: str, start: int, next_anchor_start: int) -> int:
        limit = min(len(text), start + _MAX_RESULT_DISTANCE, next_anchor_start)
        match = _CLAUSE_END_RE.search(text, start, limit)
        return match.start() if match is not None else limit

    @staticmethod
    def _candidate_spans(scope: str, *, base_offset: int) -> list[tuple[int, int]]:
        numeric = _NUMERIC_RESULT_RE.match(scope)
        if numeric is not None:
            start, end = numeric.span("value")
            while start < end and scope[start].isspace():
                start += 1
            return [(base_offset + start, base_offset + end)]
        qualitative = _QUALITATIVE_RESULT_RE.search(scope)
        if qualitative is None:
            return []
        return [(base_offset + qualitative.start(), base_offset + qualitative.end())]

    @staticmethod
    def _left_candidate_spans(text: str, anchor_start: int, previous_anchor_end: int) -> list[tuple[int, int]]:
        scope_start = max(previous_anchor_end, anchor_start - 40)
        scope = text[scope_start:anchor_start]
        last_boundary = None
        for boundary in _CLAUSE_END_RE.finditer(scope):
            last_boundary = boundary
        if last_boundary is not None:
            scope_start += last_boundary.end()
            scope = scope[last_boundary.end() :]
        match = _LEFT_QUALITATIVE_RESULT_RE.search(scope)
        if match is None:
            return []
        start, end = match.span("value")
        return [(scope_start + start, scope_start + end)]


def _overlaps(span: tuple[int, int], occupied: Iterable[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in occupied)
