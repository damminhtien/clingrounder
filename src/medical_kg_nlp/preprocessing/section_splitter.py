from __future__ import annotations
import re

from medical_kg_nlp.schema.document import Section


_LINE_RE = re.compile(r"^.*$", flags=re.MULTILINE)
_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s*[\.)]\s*")
_SEPARATOR_RE = re.compile(r"\s*(?::|：)\s*")

_HEADING_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Thuốc trước khi nhập viện", ("thuốc trước khi nhập viện lần này", "thuốc trước khi nhập viện")),
    ("Các bệnh lý mạn tính", ("các bệnh lý mãn tính", "các bệnh lý mạn tính", "bệnh lý mãn tính", "bệnh lý mạn tính")),
    ("Tiền sử bệnh nội khoa", ("tiền sử bệnh nội khoa",)),
    ("Tiền sử bệnh hiện tại", ("tiền sử bệnh hiện tại",)),
    ("Bệnh sử hiện tại", ("bệnh sử hiện tại", "lịch sử bệnh hiện tại")),
    ("Tiền sử bệnh", ("tiền sử bệnh",)),
    ("Triệu chứng hiện tại", ("các triệu chứng hiện tại", "triệu chứng hiện tại", "triệu chứng khi đến")),
    ("Các sự kiện trước khi nhập viện", ("các sự kiện trước khi nhập viện", "diễn biến trước khi nhập viện")),
    ("Đặc điểm triệu chứng", ("đặc điểm triệu chứng khi khám tại khoa cấp cứu", "đặc điểm triệu chứng")),
    ("Đánh giá tại bệnh viện", ("đánh giá tại bệnh viện",)),
    ("Kết quả khám lâm sàng", ("kết quả khám lâm sàng",)),
    ("Kết quả xét nghiệm", ("kết quả xét nghiệm",)),
    ("Kết quả chẩn đoán hình ảnh", ("kết quả chẩn đoán hình ảnh",)),
    ("Các kết quả chẩn đoán khác", ("các kết quả chẩn đoán khác", "các phát hiện chẩn đoán khác")),
    ("Điều trị", ("các thuốc đã thực hiện", "điều trị")),
    ("Lý do nhập viện", ("lý do nhập viện", "lý do khám bệnh")),
)


def split_sections(text: str) -> list[Section]:
    headings = _section_headings(text)
    if not headings:
        return [Section(title="Document", span=(0, len(text)), text=text)]

    sections: list[Section] = []
    for index, (title, heading_start, content_start) in enumerate(headings):
        content_end = headings[index + 1][1] if index + 1 < len(headings) else len(text)
        if content_start > content_end:
            content_start = heading_start
        section_text = text[content_start:content_end]
        sections.append(Section(title=title, span=(content_start, content_end), text=section_text))
    return sections


def _section_headings(text: str) -> list[tuple[str, int, int]]:
    headings: list[tuple[str, int, int]] = []
    for match in _LINE_RE.finditer(text):
        line = match.group(0)
        if not line.strip():
            continue
        heading = _match_heading(line)
        if heading is None:
            continue
        title, content_column = heading
        headings.append((title, match.start(), match.start() + content_column))
    return _dedupe_headings(headings)


def _match_heading(line: str) -> tuple[str, int] | None:
    prefix_match = _NUMBER_PREFIX_RE.match(line)
    content_offset = prefix_match.end() if prefix_match else 0
    candidate = line[content_offset:]
    stripped_left = len(candidate) - len(candidate.lstrip())
    search_start = content_offset + stripped_left
    searchable = candidate.lstrip()
    normalized = searchable.lower()
    for canonical, aliases in _HEADING_ALIASES:
        for alias in aliases:
            if not _matches_alias_boundary(normalized, alias):
                continue
            alias_end = search_start + len(alias)
            separator = _SEPARATOR_RE.match(line, alias_end)
            if separator is not None:
                return canonical, separator.end()
            remainder = line[alias_end:]
            if remainder.strip():
                return canonical, alias_end
            return canonical, len(line)
    return None


def _matches_alias_boundary(text: str, alias: str) -> bool:
    if not text.startswith(alias):
        return False
    if len(text) == len(alias):
        return True
    next_char = text[len(alias)]
    return next_char.isspace() or next_char in {":", "："}


def _dedupe_headings(headings: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    deduped: list[tuple[str, int, int]] = []
    previous_start = -1
    for heading in headings:
        if heading[1] == previous_start:
            deduped[-1] = heading
            continue
        deduped.append(heading)
        previous_start = heading[1]
    return deduped
