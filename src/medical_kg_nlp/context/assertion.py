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


class AssertionClassifier:
    def classify(self, entity: EntityAnnotation, sentence: Sentence | None = None) -> AssertionStatus:
        if entity.type in {EntityType.LAB_RESULT, EntityType.LAB_TEST}:
            return AssertionStatus.PRESENT
        sentence_text = sentence.text.lower() if sentence else ""
        entity_start = entity.span[0] - sentence.span[0] if sentence else 0
        entity_end = entity.span[1] - sentence.span[0] if sentence else entity_start
        left_context = self._local_left_context(sentence_text, max(entity_start, 0))
        right_context = self._local_right_context(sentence_text, max(entity_end, 0))

        if self._contains(left_context, FAMILY_CUES):
            return AssertionStatus.FAMILY
        if self._contains(left_context, POSSIBLE_CUES) or self._contains(right_context, POSSIBLE_CUES):
            return AssertionStatus.POSSIBLE
        if self._contains(left_context, NEGATION_CUES):
            return AssertionStatus.NEGATED
        if self._contains(left_context, HISTORICAL_CUES):
            return AssertionStatus.HISTORICAL
        if self._contains(left_context, PLANNED_CUES):
            return AssertionStatus.PLANNED
        if self._contains(left_context, RESOLVED_CUES):
            return AssertionStatus.RESOLVED
        if sentence and sentence.section_title:
            prior = SECTION_PRIORS.get(sentence.section_title.lower().strip())
            if prior is not None:
                return prior
        return AssertionStatus.PRESENT

    @staticmethod
    def _local_left_context(text: str, entity_start: int) -> str:
        left_context = text[:entity_start]
        last_boundary_end = 0
        for match in _CLAUSE_BOUNDARY_RE.finditer(left_context):
            last_boundary_end = match.end()
        return left_context[last_boundary_end:]

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
