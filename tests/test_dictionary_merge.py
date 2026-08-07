"""Concept merge tests protect canonical terminology fields from source-order loss."""

from __future__ import annotations

import pytest

from clingrounder.dictionaries.merge import merge_concept_entries
from clingrounder.dictionaries.synonym_table import ConceptEntry
from clingrounder.schema.types import CodeSystem, EntityType


def test_merge_concepts_preserves_primary_and_collects_lexical_variants() -> None:
    primary = _entry(canonical_name="metformin", aliases=("Glucophage",), source="rxnorm")
    secondary = _entry(
        canonical_name="metformin hydrochloride",
        aliases=("metformin HCl",),
        source="reviewed",
    )

    merged = merge_concept_entries((primary, secondary))

    assert len(merged) == 1
    assert merged[0].canonical_name == "metformin"
    assert merged[0].aliases == (
        "Glucophage",
        "metformin hydrochloride",
        "metformin HCl",
    )
    assert merged[0].source == "rxnorm|reviewed"


def test_merge_concepts_rejects_identity_and_structured_conflicts() -> None:
    primary = _entry(canonical_name="metformin")
    with pytest.raises(ValueError, match="Conflicting code"):
        merge_concept_entries((primary, _entry(canonical_name="other", code="999")))
    with pytest.raises(ValueError, match="Conflicting strength"):
        merge_concept_entries(
            (
                primary,
                _entry(canonical_name="metformin", strength="500 MG"),
                _entry(canonical_name="metformin", strength="850 MG"),
            )
        )


def _entry(
    *,
    canonical_name: str,
    code: str = "6809",
    aliases: tuple[str, ...] = (),
    source: str = "test",
    strength: str | None = None,
) -> ConceptEntry:
    return ConceptEntry(
        concept_id="RX:6809",
        code=code,
        code_system=CodeSystem.RXNORM,
        canonical_name=canonical_name,
        semantic_type=EntityType.DRUG,
        aliases=aliases,
        source=source,
        rxnorm_id="6809",
        rxnorm_tty="IN",
        ingredient="metformin",
        strength=strength,
    )
