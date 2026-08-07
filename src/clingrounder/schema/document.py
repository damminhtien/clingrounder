from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    title: str
    span: tuple[int, int]
    text: str
    category: str | None = None
    heading_span: tuple[int, int] | None = None
    parent_title: str | None = None
    rule_id: str | None = None

    def __post_init__(self) -> None:
        _validate_span(self.span, "section")
        if not self.title.strip() or not self.text:
            raise ValueError("Section title and text must be non-empty")
        if self.heading_span is not None:
            _validate_span(self.heading_span, "section heading")

    def validate_offsets(self, source_text: str) -> None:
        _validate_source_span(self.span, self.text, source_text, "section")
        if self.heading_span is not None:
            start, end = self.heading_span
            if not 0 <= start < end <= len(source_text):
                raise ValueError(f"Invalid section heading span {self.heading_span}")


@dataclass(frozen=True)
class Sentence:
    span: tuple[int, int]
    text: str
    section_title: str | None = None

    def __post_init__(self) -> None:
        _validate_span(self.span, "sentence")
        if not self.text:
            raise ValueError("Sentence text must be non-empty")

    def validate_offsets(self, source_text: str) -> None:
        _validate_source_span(self.span, self.text, source_text, "sentence")


@dataclass
class ClinicalDocument:
    document_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("ClinicalDocument document_id must be non-empty")
        if not self.text:
            raise ValueError("ClinicalDocument text must be non-empty")
        if any(not key.strip() or not isinstance(value, str) for key, value in self.metadata.items()):
            raise ValueError("ClinicalDocument metadata keys and values must be strings")


def _validate_span(span: tuple[int, int], label: str) -> None:
    start, end = span
    if not 0 <= start < end:
        raise ValueError(f"{label} span must satisfy 0 <= start < end: {span}")


def _validate_source_span(
    span: tuple[int, int],
    value: str,
    source_text: str,
    label: str,
) -> None:
    start, end = span
    if not 0 <= start < end <= len(source_text):
        raise ValueError(f"Invalid {label} span {span}")
    if source_text[start:end] != value:
        raise ValueError(f"{label} text does not match source span {span}")
