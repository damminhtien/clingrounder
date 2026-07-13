from __future__ import annotations

import re
from dataclasses import dataclass

from medical_kg_nlp.schema.annotation import MedicationComponent, MedicationMention


_MAX_EXTENSION_CHARS = 128
_STOP_RE = re.compile(
    r"\s*(?:cho|vì|do|để|không|nhưng|tuy nhiên|with|for|due\s+to|because)\b",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class _ComponentPattern:
    kind: str
    pattern: re.Pattern[str]
    requires_prior_component: bool = False


_COMPONENT_PATTERNS = (
    _ComponentPattern(
        "strength",
        re.compile(
            r"\s*(?:,?\s*)?\d+(?:[.,]\d+)?(?:\s*/\s*\d+(?:[.,]\d+)?)?\s*"
            r"(?:mg|g|gram|mcg|microgram|ml|m[eE]q|iu|u|đơn\s+vị|units?)"
            r"(?:\s*/\s*(?:ml|l|kg|ngày|day|lần|dose))?",
            re.IGNORECASE | re.UNICODE,
        ),
    ),
    _ComponentPattern(
        "dose_form",
        re.compile(
            r"\s*(?:viên\s+nang|viên\s+nén|dung\s+dịch|khí\s+dung|"
            r"capsules?|tablets?|injections?|solutions?|inhalers?|nebulizers?|nebs?)\b",
            re.IGNORECASE | re.UNICODE,
        ),
    ),
    _ComponentPattern(
        "route",
        re.compile(
            r"\s*(?:po|p\.o\.|iv|i\.v\.|im|sc|sl|uống|đường\s+uống|"
            r"tiêm\s+tĩnh\s+mạch|truyền\s+tĩnh\s+mạch|tĩnh\s+mạch|hít|xịt|dán|nhỏ)\b",
            re.IGNORECASE | re.UNICODE,
        ),
    ),
    _ComponentPattern(
        "frequency",
        re.compile(
            r"\s*(?:bid|tid|qid|qhs|qd|q\s*\d+\s*h|daily|once|prn|hằng\s+ngày|"
            r"hàng\s+ngày|mỗi\s+ngày|mỗi\s+\d+\s*(?:giờ|phút)|"
            r"every\s+\d+\s*(?:hours?|minutes?)|\d+\s*lần\s*/\s*ngày)\b",
            re.IGNORECASE | re.UNICODE,
        ),
    ),
    _ComponentPattern("dosage", re.compile(r"\s*x\s*\d+\b", re.IGNORECASE)),
    _ComponentPattern(
        "dosage",
        re.compile(r"\s+\d+\s*(?:lần|liều|viên|tablets?|capsules?|puffs?)\b", re.IGNORECASE),
    ),
    _ComponentPattern(
        "duration",
        re.compile(
            r"\s+trong\s+\d+(?:[.,]\d+)?\s*(?:ngày|day|days|tuần|weeks?|tháng|months?)\b",
            re.IGNORECASE | re.UNICODE,
        ),
        requires_prior_component=True,
    ),
    _ComponentPattern(
        "transition",
        re.compile(
            r"\s*,?\s*(?:sau\s+đó|then|rồi)(?:\s+giảm\s+xuống)?\s+"
            r"\d+(?:[.,]\d+)?\s*(?:mg|g|gram|mcg|microgram|ml|iu|u|đơn\s+vị|units?)"
            r"(?:\s*/\s*(?:ngày|day|lần|dose))?",
            re.IGNORECASE | re.UNICODE,
        ),
        requires_prior_component=True,
    ),
    _ComponentPattern(
        "context",
        re.compile(r"\s+tại\s+nhà\b", re.IGNORECASE | re.UNICODE),
        requires_prior_component=True,
    ),
    _ComponentPattern(
        "context",
        re.compile(r"\s*\([^)\n\r]{1,50}\)"),
        requires_prior_component=True,
    ),
)


class MedicationMentionParser:
    def parse(self, source_text: str, drug_span: tuple[int, int]) -> MedicationMention:
        start, end = drug_span
        if start < 0 or end <= start or end > len(source_text):
            return MedicationMention(drug_span=drug_span, full_span=drug_span)

        limit = min(len(source_text), end + _MAX_EXTENSION_CHARS)
        cursor = end
        components: list[MedicationComponent] = []
        while cursor < limit:
            if source_text[cursor : cursor + 1] in {"\n", "\r", ";"}:
                break
            if _STOP_RE.match(source_text, cursor):
                break

            matched: tuple[_ComponentPattern, re.Match[str]] | None = None
            for component_pattern in _COMPONENT_PATTERNS:
                if component_pattern.requires_prior_component and not components:
                    continue
                match = component_pattern.pattern.match(source_text, cursor, limit)
                if match is None or match.end() <= cursor:
                    continue
                matched = component_pattern, match
                break
            if matched is None:
                break

            component_pattern, match = matched
            component_start = cursor
            while component_start < match.end() and source_text[component_start] in " ,":
                component_start += 1
            component_end = match.end()
            while component_end > component_start and source_text[component_end - 1] in " ,":
                component_end -= 1
            if component_end > component_start:
                components.append(
                    MedicationComponent(
                        kind=component_pattern.kind,
                        span=(component_start, component_end),
                    )
                )
            cursor = match.end()

        full_end = components[-1].span[1] if components else end
        return MedicationMention(
            drug_span=drug_span,
            full_span=(start, full_end),
            components=tuple(components),
        )
