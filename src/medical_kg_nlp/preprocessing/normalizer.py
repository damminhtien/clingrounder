from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.preprocessing.offset_mapping import (
    OffsetMappedText,
    collapse_whitespace_preserve_offsets,
)
from medical_kg_nlp.utils.text import normalize_for_match


NORMALIZATION_CONTRACT_VERSION = "lookup-v1"


@dataclass(frozen=True)
class NormalizationContract:
    """Versioned boundary between source text and lookup normalization.

    Clinical spans always address ``source_text``. The mapped representation is suitable for
    lookup and diagnostics only until every downstream span producer explicitly maps its output
    back through ``normalized_to_original``. Keeping that policy in one object prevents a future
    normalizer or retrieval backend from silently changing the offset coordinate system.
    """

    version: str = NORMALIZATION_CONTRACT_VERSION
    downstream_uses_source_text: bool = True

    def prepare(self, source_text: str) -> OffsetMappedText:
        mapped = collapse_whitespace_preserve_offsets(source_text)
        self.validate(mapped)
        return mapped

    def normalize_lookup_key(self, text: str, *, strip_diacritics: bool = False) -> str:
        return normalize_for_match(text, strip_diacritics=strip_diacritics)

    @staticmethod
    def validate(mapped: OffsetMappedText) -> None:
        """Reject malformed maps before they can corrupt source offsets."""

        offsets = mapped.normalized_to_original
        if len(mapped.normalized) != len(offsets):
            raise ValueError("Normalized text and offset map must have equal length")
        if any(offset < 0 or offset >= len(mapped.original) for offset in offsets):
            raise ValueError("Normalized offset map points outside source text")
        if any(left >= right for left, right in zip(offsets, offsets[1:])):
            raise ValueError("Normalized offset map must be strictly increasing")


DEFAULT_NORMALIZATION_CONTRACT = NormalizationContract()


def normalize_text_for_matching(text: str) -> str:
    return DEFAULT_NORMALIZATION_CONTRACT.normalize_lookup_key(text)
