from __future__ import annotations

import re
from dataclasses import dataclass

from clingrounder.ner.medication_mention_parser import MedicationMentionParser
from clingrounder.schema.annotation import EntityAnnotation, MedicationMention
from clingrounder.schema.types import EntityType


_ITEM_MARKER_RE = re.compile(
    r"(?m)(?:^[ \t]*[-*•][ \t]+|(?<!\S)\d{1,3}[.)][ \t]+)",
    flags=re.UNICODE,
)
_INDICATION_MARKER_RE = re.compile(
    r"\s+(?:điều\s+trị|cho|for)\s+",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class MedicationListItem:
    medication_span: tuple[int, int]
    indication_span: tuple[int, int] | None


class MedicationListParser:
    """Parse medication-list structure without owning clinical vocabulary."""

    def __init__(self) -> None:
        self.mention_parser = MedicationMentionParser()

    def items(self, source_text: str) -> tuple[MedicationListItem, ...]:
        """Parse numbered, parenthesized, bulleted, and inline medication list items."""
        markers = list(_ITEM_MARKER_RE.finditer(source_text))
        items: list[MedicationListItem] = []
        for index, marker in enumerate(markers):
            body_start = marker.end()
            next_marker_start = (
                markers[index + 1].start() if index + 1 < len(markers) else len(source_text)
            )
            line_breaks = [
                position
                for token in ("\r", "\n")
                if (position := source_text.find(token, body_start, next_marker_start)) >= 0
            ]
            body_end = min(line_breaks, default=next_marker_start)
            body_start, body_end = _trim_span(source_text, (body_start, body_end))
            if body_end <= body_start:
                continue
            indication_marker = _INDICATION_MARKER_RE.search(
                source_text, body_start, body_end
            )
            medication_end = indication_marker.start() if indication_marker else body_end
            _, medication_end = _trim_span(source_text, (body_start, medication_end))
            indication_span = None
            if indication_marker is not None:
                indication_span = _trim_span(
                    source_text,
                    (indication_marker.end(), body_end),
                )
                if indication_span[0] >= indication_span[1]:
                    indication_span = None
            if medication_end > body_start:
                items.append(
                    MedicationListItem(
                        medication_span=(body_start, medication_end),
                        indication_span=indication_span,
                    )
                )
        return tuple(items)

    def adjudicate(
        self, source_text: str, entities: list[EntityAnnotation]
    ) -> list[EntityAnnotation]:
        items = self.items(source_text)
        if not items:
            return entities

        # List parsing supplies a hard item boundary and indication scope. The structured
        # mention parser still owns the medication boundary: treating every clinical bullet
        # as a complete medication span absorbs narrative such as prescribing rationale.
        for entity in entities:
            if entity.type != EntityType.DRUG:
                continue
            item = next(
                (item for item in items if _contains(item.medication_span, entity.span)),
                None,
            )
            if item is None:
                continue
            parsed = self.mention_parser.parse(source_text, entity.span)
            full_end = min(parsed.full_span[1], item.medication_span[1])
            entity.medication_mention = MedicationMention(
                drug_span=entity.span,
                full_span=(entity.span[0], full_end),
                components=tuple(
                    component
                    for component in parsed.components
                    if component.span[1] <= full_end
                ),
            )
        return entities


def _contains(container: tuple[int, int], inner: tuple[int, int]) -> bool:
    return container[0] <= inner[0] and inner[1] <= container[1]


def _trim_span(text: str, span: tuple[int, int]) -> tuple[int, int]:
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
