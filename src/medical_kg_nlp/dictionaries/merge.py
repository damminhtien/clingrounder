"""Deterministically merge compatible terminology records.

Canonical terminology releases occasionally overlap with curated or mined overlays.  This module
keeps the merge policy independent from in-memory and SQLite repository implementations so both
paths enforce the same concept identity rules.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import cast

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry

__all__ = ["merge_concept_entries"]


def merge_concept_entries(entries: Iterable[ConceptEntry]) -> list[ConceptEntry]:
    """Merge repeated concept IDs without discarding aliases or structured metadata.

    The first record remains authoritative for display fields. Later compatible records contribute
    lexical variants and fill missing structured fields. Identity or structured-data conflicts fail
    closed because silently selecting the last source can produce invalid medical mappings.
    """

    by_concept_id: dict[str, ConceptEntry] = {}
    order: list[str] = []
    for entry in entries:
        current = by_concept_id.get(entry.concept_id)
        if current is None:
            by_concept_id[entry.concept_id] = entry
            order.append(entry.concept_id)
            continue
        by_concept_id[entry.concept_id] = _merge_pair(current, entry)
    return [by_concept_id[concept_id] for concept_id in order]


def _merge_pair(primary: ConceptEntry, secondary: ConceptEntry) -> ConceptEntry:
    identity_fields = ("code", "code_system", "semantic_type")
    for field_name in identity_fields:
        if getattr(primary, field_name) != getattr(secondary, field_name):
            raise ValueError(
                f"Conflicting {field_name} for concept {primary.concept_id!r}: "
                f"{getattr(primary, field_name)!r} != {getattr(secondary, field_name)!r}"
            )

    # INVARIANT: one concept cannot acquire two incompatible strengths, dose forms, or RxNorm
    # identities merely because canonical files were merged in a different order.
    strict_optional_fields = ("parent_code", "rxnorm_id", "rxnorm_tty", "dose_form", "strength")
    merged_scalars = {
        field_name: _merge_optional_scalar(primary, secondary, field_name)
        for field_name in strict_optional_fields
    }

    aliases = list(primary.aliases)
    known_names = {value.casefold().strip() for value in primary.all_names}
    for value in (
        secondary.canonical_name,
        secondary.official_name_vi,
        secondary.official_name_en,
        secondary.ingredient,
        secondary.brand_name,
        secondary.generic_name,
        *secondary.aliases,
    ):
        _append_unique(aliases, known_names, value)

    return replace(
        primary,
        aliases=tuple(aliases),
        official_name_vi=primary.official_name_vi or secondary.official_name_vi,
        official_name_en=primary.official_name_en or secondary.official_name_en,
        synonyms=_ordered_union(primary.synonyms, secondary.synonyms),
        abbreviations=_ordered_union(primary.abbreviations, secondary.abbreviations),
        parents=_ordered_union(primary.parents, secondary.parents),
        blocked_aliases=_ordered_union(primary.blocked_aliases, secondary.blocked_aliases),
        source="|".join(
            _ordered_union(
                _split_sources(primary.source),
                _split_sources(secondary.source),
            )
        ),
        ingredient=primary.ingredient or secondary.ingredient,
        brand_name=primary.brand_name or secondary.brand_name,
        generic_name=primary.generic_name or secondary.generic_name,
        parent_code=merged_scalars["parent_code"],
        rxnorm_id=merged_scalars["rxnorm_id"],
        rxnorm_tty=merged_scalars["rxnorm_tty"],
        dose_form=merged_scalars["dose_form"],
        strength=merged_scalars["strength"],
    )


def _merge_optional_scalar(
    primary: ConceptEntry,
    secondary: ConceptEntry,
    field_name: str,
) -> str | None:
    left = cast(str | None, getattr(primary, field_name))
    right = cast(str | None, getattr(secondary, field_name))
    if left is not None and right is not None and left != right:
        raise ValueError(
            f"Conflicting {field_name} for concept {primary.concept_id!r}: {left!r} != {right!r}"
        )
    return left or right


def _append_unique(output: list[str], known: set[str], value: str | None) -> None:
    if value is None:
        return
    cleaned = value.strip()
    key = cleaned.casefold()
    if not cleaned or key in known:
        return
    known.add(key)
    output.append(cleaned)


def _ordered_union(left: Iterable[str], right: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in (*left, *right):
        cleaned = value.strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return tuple(output)


def _split_sources(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split("|") if part.strip())
