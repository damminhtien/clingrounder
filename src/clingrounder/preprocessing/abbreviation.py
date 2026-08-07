from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AbbreviationExpansion:
    abbreviation: str
    expansions: tuple[str, ...]

