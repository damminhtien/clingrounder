from __future__ import annotations

import re
from dataclasses import dataclass

from medical_kg_nlp.ner.medication_mention_parser import MedicationMentionParser
from medical_kg_nlp.schema.annotation import EntityAnnotation, MedicationMention
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match


_LIST_LINE_RE = re.compile(r"(?m)^\s*\d+\.\s*(?P<body>[^\r\n]+)")
_INDICATION_MARKER_RE = re.compile(r"\s+điều\s+trị\s+", re.IGNORECASE | re.UNICODE)
_REVIEWED_INDICATIONS = (
    "sốt đau",
    "đau nhức",
    "táo bón",
    "mất ngủ",
    "lo âu",
    "ho",
)
_INDICATION_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(value) for value in _REVIEWED_INDICATIONS) + r")(?!\w)",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class MedicationListItem:
    medication_span: tuple[int, int]
    indication_span: tuple[int, int] | None


class MedicationListParser:
    def __init__(self) -> None:
        self.mention_parser = MedicationMentionParser()

    def items(self, source_text: str) -> tuple[MedicationListItem, ...]:
        items: list[MedicationListItem] = []
        for match in _LIST_LINE_RE.finditer(source_text):
            body_start, body_end = match.span("body")
            marker = _INDICATION_MARKER_RE.search(source_text, body_start, body_end)
            medication_end = marker.start() if marker is not None else body_end
            while medication_end > body_start and source_text[medication_end - 1].isspace():
                medication_end -= 1
            indication_span = None
            if marker is not None:
                indication_start = marker.end()
                while indication_start < body_end and source_text[indication_start].isspace():
                    indication_start += 1
                if indication_start < body_end:
                    indication_span = (indication_start, body_end)
            items.append(MedicationListItem((body_start, medication_end), indication_span))
        return tuple(items)

    def adjudicate(
        self, source_text: str, entities: list[EntityAnnotation]
    ) -> list[EntityAnnotation]:
        items = self.items(source_text)
        if not items:
            return entities

        indication_spans = [item.indication_span for item in items if item.indication_span]
        retained = [
            entity
            for entity in entities
            if not any(_overlaps(entity.span, span) for span in indication_spans)
        ]
        for entity in retained:
            if entity.type != EntityType.DRUG:
                continue
            item = next(
                (item for item in items if _contains(item.medication_span, entity.span)),
                None,
            )
            if item is None:
                continue
            parsed = self.mention_parser.parse(source_text, entity.span)
            entity.medication_mention = MedicationMention(
                drug_span=entity.span,
                full_span=(entity.span[0], item.medication_span[1]),
                components=tuple(
                    component
                    for component in parsed.components
                    if component.span[1] <= item.medication_span[1]
                ),
            )

        for indication_span in indication_spans:
            for match in _INDICATION_RE.finditer(
                source_text, indication_span[0], indication_span[1]
            ):
                span = match.span()
                retained.append(
                    EntityAnnotation(
                        id="",
                        span=span,
                        text=source_text[span[0] : span[1]],
                        normalized_text=normalize_for_match(source_text[span[0] : span[1]]),
                        type=EntityType.SYMPTOM,
                        assertion=AssertionStatus.PRESENT,
                        code_system=CodeSystem.NONE,
                        confidence=0.98,
                    )
                )
        return retained


def _contains(container: tuple[int, int], inner: tuple[int, int]) -> bool:
    return container[0] <= inner[0] and inner[1] <= container[1]


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
