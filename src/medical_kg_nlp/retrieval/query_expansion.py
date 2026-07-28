"""Bounded, deterministic query variants for full-terminology retrieval.

The variants in this module only remove or normalize medication administration
syntax. They do not add terminology aliases or infer a medical concept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["RetrievalQueryVariant", "build_retrieval_query_variants"]


@dataclass(frozen=True)
class RetrievalQueryVariant:
    """One lower-priority query that preserves the original mention as evidence."""

    text: str
    kind: str
    score_multiplier: float


_ATTACHED_UNIT_RE = re.compile(
    r"(?<=\d)(?=(?:mcg|µg|μg|mg|gm|g|ml|meq|iu|u)\b)",
    flags=re.IGNORECASE,
)
_GRAM_RE = re.compile(r"\bgrams?\b", flags=re.IGNORECASE)
_ORAL_ROUTE_RE = re.compile(
    r"(?<!\w)(?:p\.?o\.?|đường\s+uống|uong|uống)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_STRENGTH_RE = re.compile(
    r"(?<!\w)(?:\d+(?:[.,]\d+)?|[.,]\d+)"
    r"(?:\s*-\s*(?:\d+(?:[.,]\d+)?|[.,]\d+))?\s*"
    r"(?:mcg|µg|μg|mg|gm|g|ml|meq|iu|u)"
    r"(?:\s*/\s*(?:\d+(?:[.,]\d+)?\s*)?"
    r"(?:ml|l|kg|day|ngày|dose|lần))?(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_TOKEN_RE = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*", flags=re.UNICODE)
_COUNT_TOKEN_RE = re.compile(r"(?:x|q)?\d+(?:h|min)?", flags=re.IGNORECASE)

# Administration tokens are deliberately narrow. The core query is a fallback for
# ingredient/brand recall; product-form evidence remains in the typography variant.
_ADMINISTRATION_TOKENS = frozenset(
    {
        "bid",
        "capsule",
        "capsules",
        "daily",
        "dose",
        "doses",
        "duong",
        "đường",
        "extended",
        "frequency",
        "gm",
        "gram",
        "grams",
        "hang",
        "hằng",
        "hàng",
        "im",
        "inhaler",
        "injection",
        "injectable",
        "intramuscular",
        "intravenous",
        "iv",
        "lan",
        "lần",
        "mcg",
        "meq",
        "mg",
        "ml",
        "morning",
        "ngay",
        "ngày",
        "nebs",
        "nebulizer",
        "once",
        "oral",
        "po",
        "prn",
        "qam",
        "qd",
        "qhs",
        "qid",
        "release",
        "route",
        "sc",
        "sl",
        "solution",
        "sr",
        "suspension",
        "tablet",
        "tablets",
        "tid",
        "tinh",
        "tĩnh",
        "truyen",
        "truyền",
        "uong",
        "uống",
        "xl",
        "xr",
    }
)


def build_retrieval_query_variants(
    mention: str,
    entity_type: EntityType,
) -> tuple[RetrievalQueryVariant, ...]:
    """Return at most two medication-only query expansions.

    The first variant normalizes lexical typography used by RxNorm labels. The
    second removes dose, route, form, and frequency syntax so an ingredient or
    brand can still enter top-k.

    INVARIANT: variants never replace the source mention and never produce a code.
    SCALING: at most two additional FTS queries are emitted per drug mention.
    """

    if entity_type != EntityType.DRUG:
        return ()
    original = normalize_for_match(mention)
    if not original:
        return ()

    output: list[RetrievalQueryVariant] = []
    seen = {original}
    typography = _normalize_medication_typography(original)
    if typography not in seen:
        output.append(
            RetrievalQueryVariant(
                text=typography,
                kind="medication_typography",
                score_multiplier=0.99,
            )
        )
        seen.add(typography)

    core = _medication_core_query(typography)
    if core and core not in seen:
        output.append(
            RetrievalQueryVariant(
                text=core,
                kind="medication_core",
                score_multiplier=0.65,
            )
        )
    return tuple(output)


def _normalize_medication_typography(value: str) -> str:
    separated = _ATTACHED_UNIT_RE.sub(" ", value)
    normalized_units = _GRAM_RE.sub("gm", separated)
    normalized_route = _ORAL_ROUTE_RE.sub("oral", normalized_units)
    return " ".join(normalized_route.split())


def _medication_core_query(value: str) -> str:
    without_strength = _STRENGTH_RE.sub(" ", value)
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(without_strength):
        normalized = normalize_for_match(token)
        if not normalized:
            continue
        if normalized in _ADMINISTRATION_TOKENS:
            continue
        if _COUNT_TOKEN_RE.fullmatch(normalized):
            continue
        tokens.append(normalized)
    core = " ".join(tokens)
    return core if any(len(token) >= 3 for token in tokens) else ""
