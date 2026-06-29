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


class AssertionClassifier:
    def classify(self, entity: EntityAnnotation, sentence: Sentence | None = None) -> AssertionStatus:
        if entity.type in {EntityType.LAB_RESULT, EntityType.LAB_TEST}:
            return AssertionStatus.PRESENT
        sentence_text = sentence.text.lower() if sentence else ""
        entity_start = entity.span[0] - sentence.span[0] if sentence else 0
        left_context = sentence_text[: max(entity_start, 0)]
        full_context = sentence_text

        if self._contains(full_context, FAMILY_CUES):
            return AssertionStatus.FAMILY
        if self._contains(left_context, POSSIBLE_CUES) or self._contains(full_context, POSSIBLE_CUES):
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
    def _contains(text: str, cues: tuple[str, ...]) -> bool:
        for cue in cues:
            normalized_cue = cue.strip()
            if not normalized_cue:
                continue
            pattern = rf"(?<!\w){re.escape(normalized_cue)}(?!\w)"
            if re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE):
                return True
        return False
