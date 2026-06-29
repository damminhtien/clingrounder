from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class KGNode:
    id: str
    kind: str
    label: str


@dataclass(frozen=True)
class KGEdge:
    head: str
    tail: str
    kind: str
    weight: float = 1.0

