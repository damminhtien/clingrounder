from __future__ import annotations
import re
import unicodedata


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s/+.-]", flags=re.UNICODE)


def normalize_for_match(text: str, *, strip_diacritics: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = _PUNCT_RE.sub(" ", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    if strip_diacritics:
        normalized = strip_vietnamese_tones(normalized)
    return normalized


def strip_vietnamese_tones(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def token_set(text: str) -> set[str]:
    return set(normalize_for_match(text).split())


def text_window(text: str, span: tuple[int, int], radius: int = 60) -> str:
    start = max(0, span[0] - radius)
    end = min(len(text), span[1] + radius)
    return text[start:end]

