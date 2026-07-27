from __future__ import annotations

import re
from collections.abc import Iterable

from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match


_ATTRIBUTE_PATTERNS: tuple[tuple[EntityType, re.Pattern[str]], ...] = (
    (
        EntityType.STRENGTH,
        re.compile(
            r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:mcg|µg|μg|mg|g|m[eE]q|iu|u)"
            r"(?:\s*/\s*(?:ml|l|kg))?(?!\s*/\s*d[lL])(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        ),
    ),
    (
        EntityType.DOSAGE,
        re.compile(
            r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:viên|tablets?|capsules?|puffs?|nhát)(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        ),
    ),
    (
        EntityType.ROUTE,
        re.compile(
            r"(?<!\w)(?:tiêm\s+tĩnh\s+mạch|truyền\s+tĩnh\s+mạch|đường\s+uống|"
            r"intravenous|subcutaneous|sublingual|oral|iv|im|sc|sl|po|uống|tiêm)(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        ),
    ),
    (
        EntityType.FREQUENCY,
        re.compile(
            r"(?<!\w)(?:q\d+h|bid|tid|qid|qd|daily|hằng\s+ngày|mỗi\s+\d+\s*giờ|"
            r"\d+\s*lần\s*/\s*ngày)(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        ),
    ),
    (
        EntityType.DURATION,
        re.compile(
            r"(?<!\w)(?:trong\s+)?\d+\s*(?:ngày|tuần|tháng|days?|weeks?|months?)(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        ),
    ),
    (
        EntityType.DOSAGE_FORM,
        re.compile(
            r"(?<!\w)(?:viên\s+nang|viên\s+nén|dung\s+dịch|khí\s+dung|"
            r"capsule|tablet|injection|solution|inhaler|nebulizer)(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        ),
    ),
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[\n\r.;]")


class MedicationAttributeExtractor:
    def extract(
        self,
        text: str,
        entities: Iterable[EntityAnnotation],
        *,
        occupied: Iterable[tuple[int, int]] = (),
    ) -> list[EntityAnnotation]:
        attributes: list[EntityAnnotation] = []
        seen: set[tuple[int, int, EntityType]] = set()
        occupied_spans = list(occupied)
        drugs = [entity for entity in entities if entity.type == EntityType.DRUG]
        for drug in drugs:
            window_start, window_end = _attribute_window(text, drug.span)
            window = text[window_start:window_end]
            for entity_type, pattern in _ATTRIBUTE_PATTERNS:
                for match in pattern.finditer(window):
                    span = window_start + match.start(), window_start + match.end()
                    key = (*span, entity_type)
                    if key in seen or _overlaps(span, occupied_spans):
                        continue
                    seen.add(key)
                    occupied_spans.append(span)
                    mention = text[span[0] : span[1]]
                    attributes.append(
                        EntityAnnotation(
                            id="",
                            span=span,
                            text=mention,
                            normalized_text=normalize_for_match(mention),
                            type=entity_type,
                            assertion=AssertionStatus.UNKNOWN,
                            code_system=CodeSystem.NONE,
                            confidence=0.9,
                        )
                    )
        return sorted(
            attributes, key=lambda entity: (entity.span[0], entity.span[1], entity.type.value)
        )


def _attribute_window(text: str, drug_span: tuple[int, int]) -> tuple[int, int]:
    base_left = max(0, drug_span[0] - 48)
    left = base_left
    right = min(len(text), drug_span[1] + 96)
    for match in _CLAUSE_BOUNDARY_RE.finditer(text[base_left : drug_span[0]]):
        left = base_left + match.end()
    boundary = _CLAUSE_BOUNDARY_RE.search(text[drug_span[1] : right])
    if boundary is not None:
        right = drug_span[1] + boundary.start()
    return left, right


def _overlaps(span: tuple[int, int], occupied: Iterable[tuple[int, int]]) -> bool:
    return any(span[0] < old_end and old_start < span[1] for old_start, old_end in occupied)
