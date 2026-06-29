from __future__ import annotations
from medical_kg_nlp.utils.text import normalize_for_match


def normalize_mention(text: str) -> str:
    return normalize_for_match(text)

