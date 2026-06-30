from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medical_kg_nlp.kg.validator import KGValidator

__all__ = ["KGValidator"]


def __getattr__(name: str) -> Any:
    if name == "KGValidator":
        from medical_kg_nlp.kg.validator import KGValidator

        return KGValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
