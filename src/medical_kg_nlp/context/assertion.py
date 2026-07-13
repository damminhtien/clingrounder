from __future__ import annotations
import re
from collections.abc import Callable

from medical_kg_nlp.context.rules import (
    ASSERTION_RULE_REGISTRY,
    FAMILY_LEFT_CUES,
    FAMILY_RIGHT_CUES,
    HISTORICAL_LEFT_CUES,
    HISTORICAL_RIGHT_CUES,
    NEGATION_LEFT_CUES,
    NEGATION_RIGHT_CUES,
    PLANNED_LEFT_CUES,
    PLANNED_RIGHT_CUES,
    POSSIBLE_LEFT_CUES,
    POSSIBLE_RIGHT_CUES,
    RESOLVED_LEFT_CUES,
    RESOLVED_RIGHT_CUES,
    SECTION_PRIORS,
)
from medical_kg_nlp.schema.annotation import (
    AssertionEvidence,
    AssertionFeatures,
    EntityAnnotation,
)
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.schema.types import AssertionStatus
from medical_kg_nlp.ontology.phase1 import PHASE1_TYPE_BY_ENTITY_TYPE, section_rule_for_heading


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
    re.compile(
        r"(?<!\w)không\s+nghĩ.{0,100}có\s+khả\s+năng(?!\w)", flags=re.IGNORECASE | re.UNICODE
    ),
)
_NEGATION_COORDINATION_BOUNDARY_RE = re.compile(r"[\n.;!?]", flags=re.UNICODE)
_NEGATION_COORDINATION_BREAK_RE = re.compile(
    r"(?<!\w)(nhưng|tuy\s+nhiên|song|however|but|bệnh\s+nhân\s+có|bn\s+có|ghi\s+nhận\s+có|kèm\s+theo|có"
    r"|bệnh\s+nhân\s+bị|hậu\s+phẫu|chuyển\s+sang\s+(?:sử\s+dụng|điều\s+trị\s+bằng)|bắt\s+đầu\s+dùng"
    r"|chẩn\s+đoán|có\s+khả\s+năng|được\s+kê|được\s+chỉ\s+định)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_FAMILY_FALSE_POSITIVE_PATTERNS = (
    re.compile(
        r"(?<!\w)con\s+trai\s+phát\s+hiện\s+bệnh\s+nhân(?!\w)", flags=re.IGNORECASE | re.UNICODE
    ),
    re.compile(
        r"(?<!\w)con\s+gái\s+phát\s+hiện\s+bệnh\s+nhân(?!\w)", flags=re.IGNORECASE | re.UNICODE
    ),
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
    def classify(
        self, entity: EntityAnnotation, sentence: Sentence | None = None
    ) -> AssertionStatus:
        return self.classify_features(entity, sentence).primary()

    def classify_features(
        self,
        entity: EntityAnnotation,
        sentence: Sentence | None = None,
    ) -> AssertionFeatures:
        features, _ = self.classify_features_with_evidence(entity, sentence)
        return features

    def classify_features_with_evidence(
        self,
        entity: EntityAnnotation,
        sentence: Sentence | None = None,
    ) -> tuple[AssertionFeatures, tuple[AssertionEvidence, ...]]:
        sentence_text = sentence.text.lower() if sentence else ""
        entity_start = entity.span[0] - sentence.span[0] if sentence else 0
        entity_end = entity.span[1] - sentence.span[0] if sentence else entity_start
        left_context = self._local_left_context(sentence_text, max(entity_start, 0))
        right_context = self._local_right_context(sentence_text, max(entity_end, 0))
        section_prior = self._section_prior(entity, sentence)

        statuses: set[AssertionStatus] = set()
        evidence: list[AssertionEvidence] = []

        def add(assertion: AssertionStatus, cue: str | None, scope: str) -> None:
            if cue is None:
                return
            statuses.add(assertion)
            evidence.append(
                ASSERTION_RULE_REGISTRY.evidence(assertion, cue, scope=scope)
            )

        add(
            AssertionStatus.FAMILY,
            self._matching_family(left_context, right_context),
            "left",
        )
        add(
            AssertionStatus.FAMILY,
            self._matching_cue(right_context, FAMILY_RIGHT_CUES),
            "right",
        )
        add(AssertionStatus.NEGATED, self._matching_negation(left_context), "left")
        add(
            AssertionStatus.NEGATED,
            self._matching_cue(right_context, NEGATION_RIGHT_CUES),
            "right",
        )
        add(
            AssertionStatus.NEGATED,
            self._matching_coordinated_negation(sentence_text, max(entity_start, 0)),
            "left",
        )
        self._add_directional_evidence(
            add,
            AssertionStatus.HISTORICAL,
            left_context,
            right_context,
            HISTORICAL_LEFT_CUES,
            HISTORICAL_RIGHT_CUES,
        )
        self._add_directional_evidence(
            add,
            AssertionStatus.PLANNED,
            left_context,
            right_context,
            PLANNED_LEFT_CUES,
            PLANNED_RIGHT_CUES,
        )
        self._add_directional_evidence(
            add,
            AssertionStatus.RESOLVED,
            left_context,
            right_context,
            RESOLVED_LEFT_CUES,
            RESOLVED_RIGHT_CUES,
        )
        if section_prior is not None:
            statuses.add(section_prior)
            section_title = str(sentence.section_title).lower().strip() if sentence else ""
            evidence.append(
                ASSERTION_RULE_REGISTRY.evidence(
                    section_prior,
                    section_title,
                    scope="section_prior",
                )
            )
        self._add_directional_evidence(
            add,
            AssertionStatus.POSSIBLE,
            left_context,
            right_context,
            POSSIBLE_LEFT_CUES,
            POSSIBLE_RIGHT_CUES,
        )
        unique_evidence = {item.rule_id: item for item in evidence}
        return AssertionFeatures.from_statuses(statuses), tuple(unique_evidence.values())

    def _add_directional_evidence(
        self,
        add: Callable[[AssertionStatus, str | None, str], None],
        assertion: AssertionStatus,
        left_context: str,
        right_context: str,
        left_cues: tuple[str, ...],
        right_cues: tuple[str, ...],
    ) -> None:
        add(assertion, self._matching_cue(left_context, left_cues), "left")
        add(assertion, self._matching_cue(right_context, right_cues), "right")

    @staticmethod
    def _section_prior(
        entity: EntityAnnotation, sentence: Sentence | None
    ) -> AssertionStatus | None:
        if sentence is None or not sentence.section_title:
            return None
        title = sentence.section_title.lower().strip()
        rule = section_rule_for_heading(title)
        if rule is not None and rule.type_prior is not None:
            if PHASE1_TYPE_BY_ENTITY_TYPE.get(entity.type) != rule.type_prior:
                return None
        return SECTION_PRIORS.get(title)

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
    def _matching_cue(text: str, cues: tuple[str, ...]) -> str | None:
        for cue in cues:
            normalized_cue = cue.strip()
            if not normalized_cue:
                continue
            pattern = rf"(?<!\w){re.escape(normalized_cue)}(?!\w)"
            if re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE):
                return normalized_cue
        return None

    def _matching_negation(self, left_context: str) -> str | None:
        blocked_spans = self._pattern_spans(left_context, _NEGATION_FALSE_POSITIVE_PATTERNS)
        return self._matching_cue_outside_spans(
            left_context,
            NEGATION_LEFT_CUES,
            blocked_spans,
        )

    def _matching_coordinated_negation(
        self, sentence_text: str, entity_start: int
    ) -> str | None:
        segment_start = 0
        for match in _NEGATION_COORDINATION_BOUNDARY_RE.finditer(sentence_text[:entity_start]):
            segment_start = match.end()
        segment = sentence_text[segment_start:entity_start]
        blocked_spans = self._pattern_spans(segment, _NEGATION_FALSE_POSITIVE_PATTERNS)
        best_match: re.Match[str] | None = None
        best_cue: str | None = None
        for cue in NEGATION_LEFT_CUES:
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
                    best_cue = normalized_cue
        if best_match is None:
            return None
        scope = segment[best_match.end() :]
        if len(scope) > 120:
            return None
        if _NEGATION_COORDINATION_BREAK_RE.search(scope) is not None:
            return None
        return best_cue

    def _matching_family(self, left_context: str, right_context: str) -> str | None:
        blocked_spans = self._pattern_spans(left_context, _FAMILY_FALSE_POSITIVE_PATTERNS)
        for cue in FAMILY_LEFT_CUES:
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
                return normalized_cue
            family_scope = left_context[match.end() :] + " " + right_context
            if _FAMILY_PREDICATE_RE.search(family_scope):
                return normalized_cue
        return None

    def _matching_cue_outside_spans(
        self,
        text: str,
        cues: tuple[str, ...],
        blocked_spans: list[tuple[int, int]],
    ) -> str | None:
        for cue in cues:
            normalized_cue = cue.strip()
            if not normalized_cue:
                continue
            pattern = rf"(?<!\w){re.escape(normalized_cue)}(?!\w)"
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.UNICODE):
                if not self._is_inside_spans(match.span(), blocked_spans):
                    return normalized_cue
        return None

    @staticmethod
    def _pattern_spans(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[tuple[int, int]]:
        return [match.span() for pattern in patterns for match in pattern.finditer(text)]

    @staticmethod
    def _is_inside_spans(span: tuple[int, int], blocked_spans: list[tuple[int, int]]) -> bool:
        return any(
            blocked_start <= span[0] and span[1] <= blocked_end
            for blocked_start, blocked_end in blocked_spans
        )
