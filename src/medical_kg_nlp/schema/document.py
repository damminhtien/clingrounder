from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    title: str
    span: tuple[int, int]
    text: str


@dataclass(frozen=True)
class Sentence:
    span: tuple[int, int]
    text: str
    section_title: str | None = None


@dataclass
class ClinicalDocument:
    document_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

