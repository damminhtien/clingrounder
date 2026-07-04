from __future__ import annotations
import re

from medical_kg_nlp.context.rules import (
    FAMILY_CUES,
    HISTORICAL_CUES,
    NEGATION_CUES,
    PLANNED_CUES,
    POSSIBLE_CUES,
    RESOLVED_CUES,
    SECTION_PRIORS,
)
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.schema.types import AssertionStatus, EntityType


_CLAUSE_BOUNDARY_RE = re.compile(r"[\n.;:!?]|,\s+")
_SCOPE_RESET_RE = re.compile(
    r"(?<!\w)(?:"
    r"chuyển\s+sang\s+(?:sử\s+dụng|điều\s+trị\s+bằng)"
    r"|bắt\s+đầu\s+dùng"
    r"|chẩn\s+đoán"
    r"|được\s+kê"
    r"|được\s+chỉ\s+định"
    r")(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_NEGATION_FALSE_POSITIVE_PATTERNS = (
    re.compile(r"(?<!\w)không\s+đặc\s+hiệu(?!\w)", flags=re.IGNORECASE | re.UNICODE),
    re.compile(r"(?<!\w)không\s+loại\s+trừ(?!\w)", flags=re.IGNORECASE | re.UNICODE),
    re.compile(r"(?<!\w)không\s+thuốc\s+cản\s+quang(?!\w)", flags=re.IGNORECASE | re.UNICODE),
    re.compile(r"(?<!\w)không\s+thể\s+giữ\s+được(?!\w)", flags=re.IGNORECASE | re.UNICODE),
    re.compile(r"(?<!\w)không\s+nghĩ.{0,100}có\s+khả\s+năng(?!\w)", flags=re.IGNORECASE | re.UNICODE),
)
_NEGATION_COORDINATION_BOUNDARY_RE = re.compile(r"[\n.;!?]", flags=re.UNICODE)
_NEGATION_COORDINATION_BREAK_RE = re.compile(
    r"(?<!\w)(nhưng|tuy\s+nhiên|song|however|but|bệnh\s+nhân\s+có|bn\s+có|ghi\s+nhận\s+có|kèm\s+theo|có"
    r"|bệnh\s+nhân\s+bị|hậu\s+phẫu|chuyển\s+sang\s+(?:sử\s+dụng|điều\s+trị\s+bằng)|bắt\s+đầu\s+dùng"
    r"|chẩn\s+đoán|có\s+khả\s+năng|được\s+kê|được\s+chỉ\s+định)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_FAMILY_FALSE_POSITIVE_PATTERNS = (
    re.compile(r"(?<!\w)con\s+trai\s+phát\s+hiện\s+bệnh\s+nhân(?!\w)", flags=re.IGNORECASE | re.UNICODE),
    re.compile(r"(?<!\w)con\s+gái\s+phát\s+hiện\s+bệnh\s+nhân(?!\w)", flags=re.IGNORECASE | re.UNICODE),
)
_FAMILY_MEMBER_CUES = frozenset(
    {
        "anh",
        "bà",
        "bố",
        "brother",
        "cha",
        "chị",
        "em",
        "father",
        "mẹ",
        "mother",
        "ông",
        "parent",
        "sister",
    }
)
_FAMILY_PREDICATE_RE = re.compile(
    r"(?<!\w)(bị|mắc|có|tiền\s+sử|history\s+of|diagnosed\s+with)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)


class AssertionClassifier:
    def classify(self, entity: EntityAnnotation, sentence: Sentence | None = None) -> AssertionStatus:
        if entity.type in {EntityType.LAB_RESULT, EntityType.LAB_TEST}:
            return AssertionStatus.PRESENT
        sentence_text = sentence.text.lower() if sentence else ""
        entity_start = entity.span[0] - sentence.span[0] if sentence else 0
        entity_end = entity.span[1] - sentence.span[0] if sentence else entity_start
        left_context = self._local_left_context(sentence_text, max(entity_start, 0))
        right_context = self._local_right_context(sentence_text, max(entity_end, 0))
        section_prior = self._section_prior(sentence)

        if self._contains_family(left_context, right_context):
            return AssertionStatus.FAMILY
        if self._contains_negation(left_context):
            return AssertionStatus.NEGATED
        if self._contains_coordinated_negation(sentence_text, max(entity_start, 0)):
            return AssertionStatus.NEGATED
        if self._contains(left_context, HISTORICAL_CUES):
            return AssertionStatus.HISTORICAL
        if self._contains(left_context, PLANNED_CUES):
            return AssertionStatus.PLANNED
        if self._contains(left_context, RESOLVED_CUES):
            return AssertionStatus.RESOLVED
        if section_prior is not None:
            return section_prior
        if self._contains(left_context, POSSIBLE_CUES) or self._contains(right_context, POSSIBLE_CUES):
            return AssertionStatus.POSSIBLE
        return AssertionStatus.PRESENT

    @staticmethod
    def _section_prior(sentence: Sentence | None) -> AssertionStatus | None:
        if sentence is None or not sentence.section_title:
            return None
        return SECTION_PRIORS.get(sentence.section_title.lower().strip())

    @staticmethod
    def _local_left_context(text: str, entity_start: int) -> str:
        left_context = text[:entity_start]
        last_boundary_end = 0
        for match in _CLAUSE_BOUNDARY_RE.finditer(left_context):
            last_boundary_end = match.end()
        scoped_context = left_context[last_boundary_end:]
        last_reset_end = 0
        for match in _SCOPE_RESET_RE.finditer(scoped_context):
            last_reset_end = match.end()
        return scoped_context[last_reset_end:]

    @staticmethod
    def _local_right_context(text: str, entity_end: int) -> str:
        right_context = text[entity_end:]
        match = _CLAUSE_BOUNDARY_RE.search(right_context)
        if match is None:
            return right_context
        return right_context[: match.start()]

    @staticmethod
    def _contains(text: str, cues: tuple[str, ...]) -> bool:
        for cue in cues:
            normalized_cue = cue.strip()
            if not normalized_cue:
                continue
            pattern = rf"(?<!\w){re.escape(normalized_cue)}(?!\w)"
            if re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE):
                return True
        return False

    def _contains_negation(self, left_context: str) -> bool:
        blocked_spans = self._pattern_spans(left_context, _NEGATION_FALSE_POSITIVE_PATTERNS)
        return self._contains_outside_spans(left_context, NEGATION_CUES, blocked_spans)

    def _contains_coordinated_negation(self, sentence_text: str, entity_start: int) -> bool:
        segment_start = 0
        for match in _NEGATION_COORDINATION_BOUNDARY_RE.finditer(sentence_text[:entity_start]):
            segment_start = match.end()
        segment = sentence_text[segment_start:entity_start]
        blocked_spans = self._pattern_spans(segment, _NEGATION_FALSE_POSITIVE_PATTERNS)
        best_match: re.Match[str] | None = None
        for cue in NEGATION_CUES:
            normalized_cue = cue.strip()
            if not normalized_cue:
                continue
            pattern = rf"(?<!\w){re.escape(normalized_cue)}(?!\w)"
            for match in re.finditer(pattern, segment, flags=re.IGNORECASE | re.UNICODE):
                if self._is_inside_spans(match.span(), blocked_spans):
                    continue
                if (
                    best_match is None
                    or match.start() > best_match.start()
                    or (match.start() == best_match.start() and match.end() > best_match.end())
                ):
                    best_match = match
        if best_match is None:
            return False
        scope = segment[best_match.end() :]
        if len(scope) > 120:
            return False
        return _NEGATION_COORDINATION_BREAK_RE.search(scope) is None

    def _contains_family(self, left_context: str, right_context: str) -> bool:
        blocked_spans = self._pattern_spans(left_context, _FAMILY_FALSE_POSITIVE_PATTERNS)
        for cue in FAMILY_CUES:
            normalized_cue = cue.strip()
            if not normalized_cue:
                continue
            pattern = rf"(?<!\w){re.escape(normalized_cue)}(?!\w)"
            match = re.search(pattern, left_context, flags=re.IGNORECASE | re.UNICODE)
            if match is None:
                continue
            if self._is_inside_spans(match.span(), blocked_spans):
                continue
            if normalized_cue.lower() not in _FAMILY_MEMBER_CUES:
                return True
            family_scope = left_context[match.end() :] + " " + right_context
            if _FAMILY_PREDICATE_RE.search(family_scope):
                return True
        return False

    def _contains_outside_spans(
        self,
        text: str,
        cues: tuple[str, ...],
        blocked_spans: list[tuple[int, int]],
    ) -> bool:
        for cue in cues:
            normalized_cue = cue.strip()
            if not normalized_cue:
                continue
            pattern = rf"(?<!\w){re.escape(normalized_cue)}(?!\w)"
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.UNICODE):
                if not self._is_inside_spans(match.span(), blocked_spans):
                    return True
        return False

    @staticmethod
    def _pattern_spans(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[tuple[int, int]]:
        return [match.span() for pattern in patterns for match in pattern.finditer(text)]

    @staticmethod
    def _is_inside_spans(span: tuple[int, int], blocked_spans: list[tuple[int, int]]) -> bool:
        return any(blocked_start <= span[0] and span[1] <= blocked_end for blocked_start, blocked_end in blocked_spans)
