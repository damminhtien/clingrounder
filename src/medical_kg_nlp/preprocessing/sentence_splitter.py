from __future__ import annotations
import re

from medical_kg_nlp.schema.document import Sentence


_SENTENCE_END_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)", flags=re.UNICODE)


def split_sentences(text: str, section_title: str | None = None, base_offset: int = 0) -> list[Sentence]:
    sentences: list[Sentence] = []
    for match in _SENTENCE_END_RE.finditer(text):
        raw_start, raw_end = match.span()
        segment = match.group(0)
        leading = len(segment) - len(segment.lstrip())
        trailing = len(segment.rstrip()) - len(segment.rstrip(" \t\r\n"))
        start = raw_start + leading
        end = raw_end - trailing
        if start >= end:
            continue
        sentence_text = text[start:end]
        sentences.append(
            Sentence(
                span=(base_offset + start, base_offset + end),
                text=sentence_text,
                section_title=section_title,
            )
        )
    return sentences

