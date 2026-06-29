from __future__ import annotations
import re

from medical_kg_nlp.schema.document import Section


_SECTION_RE = re.compile(
    r"^(?P<title>[A-Za-zÀ-ỹ /_-]{3,40})(?:\:|：)\s*$",
    flags=re.MULTILINE,
)


def split_sections(text: str) -> list[Section]:
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [Section(title="Document", span=(0, len(text)), text=text)]

    sections: list[Section] = []
    for index, match in enumerate(matches):
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = match.group("title").strip()
        section_text = text[content_start:content_end]
        sections.append(Section(title=title, span=(content_start, content_end), text=section_text))
    return sections

