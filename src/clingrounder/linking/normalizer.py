from __future__ import annotations
from clingrounder.utils.text import normalize_for_match


def normalize_mention(text: str) -> str:
    return normalize_for_match(text)

