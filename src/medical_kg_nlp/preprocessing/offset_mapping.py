from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class OffsetMappedText:
    original: str
    normalized: str
    normalized_to_original: tuple[int, ...]

    def normalized_span_to_original(self, span: tuple[int, int]) -> tuple[int, int]:
        start, end = span
        if start < 0 or end < start or end > len(self.normalized):
            raise ValueError(f"Invalid normalized span {span}")
        if start == end:
            original = self.normalized_to_original[start] if start < len(self.normalized_to_original) else len(self.original)
            return (original, original)
        original_start = self.normalized_to_original[start]
        original_end = self.normalized_to_original[end - 1] + 1
        return (original_start, original_end)


def collapse_whitespace_preserve_offsets(text: str) -> OffsetMappedText:
    chars: list[str] = []
    mapping: list[int] = []
    pending_space_index: int | None = None
    previous_was_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if not previous_was_space:
                pending_space_index = index
                chars.append(" ")
                mapping.append(index)
            previous_was_space = True
            continue
        if pending_space_index is not None and not chars[-1].strip():
            mapping[-1] = pending_space_index
        pending_space_index = None
        previous_was_space = False
        chars.append(char)
        mapping.append(index)
    normalized = "".join(chars)
    leading = len(normalized) - len(normalized.lstrip())
    trailing = len(normalized.rstrip()) - len(normalized.rstrip(" "))
    end = len(normalized) - trailing if trailing else len(normalized)
    return OffsetMappedText(
        original=text,
        normalized=normalized[leading:end],
        normalized_to_original=tuple(mapping[leading:end]),
    )
