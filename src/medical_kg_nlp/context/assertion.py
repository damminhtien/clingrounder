from __future__ import annotations
import re
from collections.abc import Callable

from medical_kg_nlp.context.cue_loader import AssertionCue, AssertionRuleRegistry
from medical_kg_nlp.context.modifier_graph import (
    AssertionDecision,
    ContextGraph,
    build_context_graph,
)
from medical_kg_nlp.context.rules import ASSERTION_RULE_REGISTRY
from medical_kg_nlp.schema.annotation import (
    AssertionEvidence,
    AssertionFeatures,
    EntityAnnotation,
)
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.schema.types import AssertionStatus
from medical_kg_nlp.schema.types import EntityType
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
    def __init__(
        self,
        registry: AssertionRuleRegistry = ASSERTION_RULE_REGISTRY,
    ) -> None:
        self.registry = registry

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

        def add(rule: AssertionCue | None, scope: str) -> None:
            if rule is None:
                return
            statuses.add(rule.assertion)
            evidence.append(
                self.registry.evidence_for_rule(rule, scope=scope)
            )

        add(self._matching_family(left_context, right_context, entity.type), "left")
        add(
            self._matching_cue(
                right_context,
                AssertionStatus.FAMILY,
                "right",
                entity.type,
            ),
            "right",
        )
        add(self._matching_negation(left_context, entity.type), "left")
        add(
            self._matching_cue(
                right_context,
                AssertionStatus.NEGATED,
                "right",
                entity.type,
            ),
            "right",
        )
        add(
            self._matching_coordinated_negation(
                sentence_text,
                max(entity_start, 0),
                entity.type,
            ),
            "left",
        )
        self._add_directional_evidence(
            add,
            AssertionStatus.HISTORICAL,
            left_context,
            right_context,
            entity.type,
        )
        self._add_directional_evidence(
            add,
            AssertionStatus.PLANNED,
            left_context,
            right_context,
            entity.type,
        )
        self._add_directional_evidence(
            add,
            AssertionStatus.RESOLVED,
            left_context,
            right_context,
            entity.type,
        )
        if section_prior is not None:
            statuses.add(section_prior.assertion)
            evidence.append(self.registry.evidence_for_rule(section_prior, scope="section_prior"))
        self._add_directional_evidence(
            add,
            AssertionStatus.POSSIBLE,
            left_context,
            right_context,
            entity.type,
        )
        unique_evidence = {item.rule_id: item for item in evidence}
        return AssertionFeatures.from_statuses(statuses), tuple(unique_evidence.values())

    def classify_batch_with_graph(
        self,
        entities: list[EntityAnnotation],
        sentence: Sentence,
    ) -> tuple[dict[str, AssertionDecision], ContextGraph]:
        """Classify one sentence and retain explicit modifier-target evidence.

        SCALING: callers can process one sentence as a unit and reuse the graph
        for feature extraction or review instead of reconstructing provenance
        from entity-level flags.
        """

        decisions = {
            entity.id: self.classify_features_with_evidence(entity, sentence)
            for entity in entities
        }
        return decisions, build_context_graph(sentence, entities, decisions)

    def _add_directional_evidence(
        self,
        add: Callable[[AssertionCue | None, str], None],
        assertion: AssertionStatus,
        left_context: str,
        right_context: str,
        entity_type: EntityType,
    ) -> None:
        add(
            self._matching_cue(left_context, assertion, "left", entity_type),
            "left",
        )
        add(
            self._matching_cue(right_context, assertion, "right", entity_type),
            "right",
        )

    def _section_prior(
        self, entity: EntityAnnotation, sentence: Sentence | None
    ) -> AssertionCue | None:
        if sentence is None or not sentence.section_title:
            return None
        title = sentence.section_title.lower().strip()
        rule = section_rule_for_heading(title)
        if rule is not None and rule.type_prior is not None:
            if PHASE1_TYPE_BY_ENTITY_TYPE.get(entity.type) != rule.type_prior:
                return None
        return self.registry.section_prior(title)

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

    def _matching_cue(
        self,
        text: str,
        assertion: AssertionStatus,
        scope: str,
        entity_type: EntityType,
    ) -> AssertionCue | None:
        matches = self._rule_matches(
            text,
            assertion,
            scope=scope,
            entity_type=entity_type,
        )
        return matches[0][1] if matches else None

    def _matching_negation(
        self,
        left_context: str,
        entity_type: EntityType,
    ) -> AssertionCue | None:
        blocked_spans = self._pattern_spans(left_context, _NEGATION_FALSE_POSITIVE_PATTERNS)
        matches = self._rule_matches(
            left_context,
            AssertionStatus.NEGATED,
            scope="left",
            entity_type=entity_type,
            blocked_spans=blocked_spans,
        )
        return matches[0][1] if matches else None

    def _matching_coordinated_negation(
        self,
        sentence_text: str,
        entity_start: int,
        entity_type: EntityType,
    ) -> AssertionCue | None:
        segment_start = 0
        for match in _NEGATION_COORDINATION_BOUNDARY_RE.finditer(sentence_text[:entity_start]):
            segment_start = match.end()
        segment = sentence_text[segment_start:entity_start]
        blocked_spans = self._pattern_spans(segment, _NEGATION_FALSE_POSITIVE_PATTERNS)
        for _, rule, match in self._rule_matches(
            segment,
            AssertionStatus.NEGATED,
            scope="left",
            entity_type=entity_type,
            blocked_spans=blocked_spans,
        ):
            if _NEGATION_COORDINATION_BREAK_RE.search(segment[match.end() :]) is None:
                return rule
        return None

    def _matching_family(
        self,
        left_context: str,
        right_context: str,
        entity_type: EntityType,
    ) -> AssertionCue | None:
        blocked_spans = self._pattern_spans(left_context, _FAMILY_FALSE_POSITIVE_PATTERNS)
        for _, rule, match in self._rule_matches(
            left_context,
            AssertionStatus.FAMILY,
            scope="left",
            entity_type=entity_type,
            blocked_spans=blocked_spans,
        ):
            if rule.cue.casefold() not in _FAMILY_MEMBER_CUES:
                return rule
            family_scope = left_context[match.end() :] + " " + right_context
            if _FAMILY_PREDICATE_RE.search(family_scope):
                return rule
        return None

    def _rule_matches(
        self,
        text: str,
        assertion: AssertionStatus,
        *,
        scope: str,
        entity_type: EntityType,
        blocked_spans: list[tuple[int, int]] | None = None,
    ) -> list[tuple[tuple[int, int, int, str], AssertionCue, re.Match[str]]]:
        matches: list[
            tuple[tuple[int, int, int, str], AssertionCue, re.Match[str]]
        ] = []
        blocked_spans = blocked_spans or []
        for rule in self.registry.rules(assertion, scope=scope):
            if not rule.applies_to(entity_type):
                continue
            pattern = rf"(?<!\w){re.escape(rule.cue.strip())}(?!\w)"
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.UNICODE):
                if self._is_inside_spans(match.span(), blocked_spans):
                    continue
                if self._terminated(rule, text, match, scope):
                    continue
                distance = len(text) - match.end() if scope == "left" else match.start()
                if distance > rule.max_distance:
                    continue
                # Priority is an explicit policy decision. Distance and cue length only break ties,
                # followed by rule_id for reproducibility across resource file reorderings.
                key = (-rule.priority, distance, -len(rule.cue), rule.rule_id)
                matches.append((key, rule, match))
        return sorted(matches, key=lambda item: item[0])

    @staticmethod
    def _terminated(
        rule: AssertionCue,
        text: str,
        match: re.Match[str],
        scope: str,
    ) -> bool:
        if not rule.termination_cues:
            return False
        between = text[match.end() :] if scope == "left" else text[: match.start()]
        return any(
            re.search(
                rf"(?<!\w){re.escape(cue)}(?!\w)",
                between,
                flags=re.IGNORECASE | re.UNICODE,
            )
            is not None
            for cue in rule.termination_cues
        )

    @staticmethod
    def _pattern_spans(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[tuple[int, int]]:
        return [match.span() for pattern in patterns for match in pattern.finditer(text)]

    @staticmethod
    def _is_inside_spans(span: tuple[int, int], blocked_spans: list[tuple[int, int]]) -> bool:
        return any(
            blocked_start <= span[0] and span[1] <= blocked_end
            for blocked_start, blocked_end in blocked_spans
        )
