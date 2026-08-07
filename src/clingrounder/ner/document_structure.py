"""Raw-offset document structure shared by deterministic NER proposal sources."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from clingrounder.utils.text import normalize_for_match

__all__ = [
    "DocumentGenre",
    "DocumentStructure",
    "DocumentStructureAnalyzer",
    "LineRegion",
    "SectionKind",
    "SectionRegion",
    "classify_section_heading_label",
]


class DocumentGenre(StrEnum):
    """Coarse document style used as evidence, never as an extraction veto."""

    CLINICAL_NOTE = "clinical_note"
    EDUCATIONAL = "educational"
    LAB_TABLE = "lab_table"
    MEDICATION_LIST = "medication_list"
    QUESTION_ANSWER = "question_answer"
    UNKNOWN = "unknown"


class SectionKind(StrEnum):
    """Stable structural section categories understood by rule adapters."""

    DIAGNOSIS = "diagnosis"
    HISTORY = "history"
    IMAGING = "imaging"
    LABORATORY = "laboratory"
    MEDICATION = "medication"
    NEUTRAL = "neutral"
    SYMPTOM = "symptom"


@dataclass(frozen=True, slots=True)
class LineRegion:
    """One physical source line with its list-item content boundary."""

    span: tuple[int, int]
    content_start: int
    is_list_item: bool

    def contains(self, boundary: int) -> bool:
        return self.span[0] <= boundary <= self.span[1]


@dataclass(frozen=True, slots=True)
class SectionRegion:
    """One heading-owned source region."""

    span: tuple[int, int]
    heading_span: tuple[int, int]
    kind: SectionKind

    def contains(self, boundary: int) -> bool:
        return self.span[0] <= boundary < self.span[1]


@dataclass(frozen=True, slots=True)
class DocumentStructure:
    """Immutable structural evidence for one raw source document."""

    text_length: int
    genre: DocumentGenre
    lines: tuple[LineRegion, ...]
    sections: tuple[SectionRegion, ...]

    def line_at(self, boundary: int) -> LineRegion | None:
        """Return the physical line containing ``boundary``."""

        if not 0 <= boundary <= self.text_length:
            raise ValueError(f"Boundary {boundary} is outside source length {self.text_length}")
        return next((line for line in self.lines if line.contains(boundary)), None)

    def section_at(self, boundary: int) -> SectionRegion | None:
        """Return the nearest explicit section that owns ``boundary``."""

        if not 0 <= boundary <= self.text_length:
            raise ValueError(f"Boundary {boundary} is outside source length {self.text_length}")
        return next(
            (section for section in reversed(self.sections) if section.contains(boundary)),
            None,
        )

    def starts_list_item(self, boundary: int) -> bool:
        """Return whether a span begins at the first content character of a list item."""

        line = self.line_at(boundary)
        return bool(line is not None and line.is_list_item and line.content_start == boundary)


_LIST_PREFIX_RE = re.compile(
    r"^[ \t]*(?:(?:[-*+•]+)[ \t]+|(?:\d{1,3}|[ivxlcdm]{1,8})[.)][ \t]+)",
    flags=re.IGNORECASE | re.UNICODE,
)
_QUESTION_ANSWER_RE = re.compile(
    r"^[ \t]*(?:câu[ \t]+hỏi|hỏi|đáp[ \t]+án|trả[ \t]+lời|question|answer)"
    r"[ \t]*(?::|-)",
    flags=re.IGNORECASE | re.MULTILINE | re.UNICODE,
)
_EDUCATIONAL_RE = re.compile(
    r"(?<!\w)(?:là[ \t]+gì|tại[ \t]+sao|giải[ \t]+thích|cơ[ \t]+chế|"
    r"nguyên[ \t]+nhân|(?:biểu[ \t]+hiện|triệu[ \t]+chứng)"
    r"[ \t]+thường[ \t]+gặp)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)

_SECTION_PATTERNS: tuple[tuple[SectionKind, re.Pattern[str]], ...] = (
    (
        SectionKind.IMAGING,
        re.compile(
            r"^(?:kết[ \t]+quả[ \t]+)?(?:chẩn[ \t]+đoán[ \t]+)?hình[ \t]+ảnh\b"
        ),
    ),
    (
        SectionKind.LABORATORY,
        re.compile(
            r"^(?:(?:kết[ \t]+quả[ \t]+)?"
            r"(?:(?:xét[ \t]*nghiệm|phòng[ \t]+thí[ \t]+nghiệm)"
            r"(?:[ \t]+(?:(?:&|và)[ \t]+)?cận[ \t]+lâm[ \t]+sàng)?|"
            r"cận[ \t]+lâm[ \t]+sàng)"
            r"(?:[ \t]+đã[ \t]+có)?|laboratory|labs?)\b"
        ),
    ),
    (
        SectionKind.SYMPTOM,
        re.compile(
            r"^(?:các[ \t]+)?triệu[ \t]+chứng"
            r"(?:[ \t]+(?:hiện[ \t]+tại|khi[ \t]+nhập[ \t]+viện|"
            r"kèm[ \t]+theo|liên[ \t]+quan))?\b|"
            r"^đặc[ \t]+điểm[ \t]+triệu[ \t]+chứng\b"
        ),
    ),
    (
        SectionKind.DIAGNOSIS,
        re.compile(
            r"^chẩn[ \t]+đoán\b|^(?:các[ \t]+)?bệnh[ \t]+lý"
            r"(?:[ \t]+nội[ \t]+khoa)?[ \t]+m(?:ạ|ã)n[ \t]+tính\b"
        ),
    ),
    (
        SectionKind.HISTORY,
        re.compile(
            r"^tiền[ \t]+sử(?:[ \t]+bệnh(?:[ \t]+nội[ \t]+khoa|[ \t]+hiện[ \t]+tại)?)?\b|"
            r"^bệnh[ \t]+sử\b"
        ),
    ),
    (
        SectionKind.MEDICATION,
        re.compile(
            r"^(?:danh[ \t]+sách[ \t]+)?thuốc\b|^điều[ \t]+trị\b|"
            r"^medications?\b"
        ),
    ),
    (
        SectionKind.NEUTRAL,
        re.compile(
            r"^lý[ \t]+do[ \t]+(?:nhập|vào)[ \t]+viện\b|"
            r"^diễn[ \t]+biến[ \t]+bệnh\b|"
            r"^(?:các[ \t]+)?sự[ \t]+kiện[ \t]+trước[ \t]+khi[ \t]+nhập[ \t]+viện\b|"
            r"^tình[ \t]+trạng[ \t]+ngay[ \t]+trước[ \t]+khi[ \t]+nhập[ \t]+viện\b|"
            r"^đánh[ \t]+giá[ \t]+tại[ \t]+bệnh[ \t]+viện\b|"
            r"^(?:các[ \t]+)?thủ[ \t]+thuật\b|^kế[ \t]+hoạch\b"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class DocumentStructureAnalyzer:
    """Analyze sections and coarse genre without transforming source text.

    INVARIANT: all returned boundaries address the original string. Normalization is used only
    to classify heading text.
    """

    def analyze(self, source_text: str) -> DocumentStructure:
        lines = _line_regions(source_text)
        headings = tuple(
            heading
            for line in lines
            if (heading := _heading_for_line(source_text, line)) is not None
        )
        sections = tuple(
            SectionRegion(
                span=(
                    heading[0][0],
                    headings[index + 1][0][0]
                    if index + 1 < len(headings)
                    else len(source_text),
                ),
                heading_span=heading[0],
                kind=heading[1],
            )
            for index, heading in enumerate(headings)
        )
        return DocumentStructure(
            text_length=len(source_text),
            genre=_classify_genre(source_text, lines, sections),
            lines=lines,
            sections=sections,
        )


def _line_regions(source_text: str) -> tuple[LineRegion, ...]:
    if not source_text:
        return ()
    lines: list[LineRegion] = []
    cursor = 0
    for physical_line in source_text.splitlines(keepends=True):
        visible = physical_line.rstrip("\r\n")
        end = cursor + len(visible)
        prefix = _LIST_PREFIX_RE.match(visible)
        leading = len(visible) - len(visible.lstrip(" \t"))
        lines.append(
            LineRegion(
                span=(cursor, end),
                content_start=cursor + (prefix.end() if prefix is not None else leading),
                is_list_item=prefix is not None,
            )
        )
        cursor += len(physical_line)
    if cursor < len(source_text):
        trailing = source_text[cursor:]
        prefix = _LIST_PREFIX_RE.match(trailing)
        leading = len(trailing) - len(trailing.lstrip(" \t"))
        lines.append(
            LineRegion(
                span=(cursor, len(source_text)),
                content_start=cursor + (prefix.end() if prefix is not None else leading),
                is_list_item=prefix is not None,
            )
        )
    return tuple(lines)


def _heading_for_line(
    source_text: str,
    line: LineRegion,
) -> tuple[tuple[int, int], SectionKind] | None:
    start, end = line.span
    raw = source_text[start:end].strip()
    if not raw:
        return None
    kind = classify_section_heading_label(raw)
    return (line.span, kind) if kind is not None else None


def classify_section_heading_label(label: str) -> SectionKind | None:
    """Classify a complete structural label without changing source offsets."""

    normalized = normalize_for_match(label.strip())
    normalized = re.sub(
        r"^(?:[-*+•]+|(?:\d{1,3}|[ivxlcdm]{1,8})[.)])\s*",
        "",
        normalized,
    )
    for kind, pattern in _SECTION_PATTERNS:
        match = pattern.match(normalized)
        if match is None:
            continue
        # A heading must terminate or introduce content with a colon. This prevents prose such as
        # "triệu chứng thường gặp là..." from opening an unbounded section.
        remainder = normalized[match.end() :]
        if remainder and not remainder.lstrip().startswith(":"):
            continue
        return kind
    return None


def _classify_genre(
    source_text: str,
    lines: tuple[LineRegion, ...],
    sections: tuple[SectionRegion, ...],
) -> DocumentGenre:
    normalized = normalize_for_match(source_text)
    if len(_QUESTION_ANSWER_RE.findall(source_text)) >= 2 or source_text.count("?") >= 2:
        return DocumentGenre.QUESTION_ANSWER
    section_kinds = {section.kind for section in sections}
    list_item_count = sum(line.is_list_item for line in lines)
    if len(sections) >= 2:
        return DocumentGenre.CLINICAL_NOTE
    if SectionKind.MEDICATION in section_kinds and list_item_count >= 2:
        return DocumentGenre.MEDICATION_LIST
    if SectionKind.LABORATORY in section_kinds and list_item_count >= 2:
        return DocumentGenre.LAB_TABLE
    if _EDUCATIONAL_RE.search(normalized):
        return DocumentGenre.EDUCATIONAL
    return DocumentGenre.UNKNOWN
