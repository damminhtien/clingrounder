"""Tests for synonym-aligned dense index contracts and hard-negative mining."""

from __future__ import annotations

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.retrieval.synonym_index import (
    InMemorySynonymVectorIndex,
    build_synonym_vector_records,
    fingerprint_terminology_entries,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.training.synonym_retrieval import build_synonym_retrieval_examples
from medical_kg_nlp.training.terminology_pairs import build_terminology_synonym_pairs


def test_synonym_vectors_deduplicate_concepts_after_type_filtered_search() -> None:
    disease = _entry(
        "ICD:I10",
        "I10",
        EntityType.DISEASE,
        CodeSystem.ICD10,
        "tăng huyết áp",
        aliases=("cao huyết áp",),
    )
    drug = _entry(
        "RX:1",
        "1",
        EntityType.DRUG,
        CodeSystem.RXNORM,
        "aspirin",
    )
    records = build_synonym_vector_records((disease, drug), _Encoder())
    index = InMemorySynonymVectorIndex(records)

    hits = index.search(
        (1.0, 0.0),
        entity_type=EntityType.DISEASE,
        code_systems=(CodeSystem.ICD10,),
        limit=1,
    )

    assert [hit.concept_id for hit in hits] == ["ICD:I10"]


def test_terminology_fingerprint_is_order_independent_and_surface_sensitive() -> None:
    first = _entry(
        "ICD:I10",
        "I10",
        EntityType.DISEASE,
        CodeSystem.ICD10,
        "tăng huyết áp",
    )
    second = _entry(
        "ICD:E11",
        "E11",
        EntityType.DISEASE,
        CodeSystem.ICD10,
        "đái tháo đường",
    )

    assert fingerprint_terminology_entries((first, second)) == fingerprint_terminology_entries(
        (second, first)
    )
    changed = _entry(
        "ICD:I10",
        "I10",
        EntityType.DISEASE,
        CodeSystem.ICD10,
        "cao huyết áp",
    )
    assert fingerprint_terminology_entries((first,)) != fingerprint_terminology_entries(
        (changed,)
    )


def test_hard_negative_mining_prioritizes_same_ingredient_wrong_strength() -> None:
    metformin_500 = _drug("RX:500", "500", "metformin 500 mg tablet", "500 mg")
    metformin_850 = _drug("RX:850", "850", "metformin 850 mg tablet", "850 mg")
    aspirin = _drug("RX:A", "A", "aspirin 81 mg tablet", "81 mg", ingredient="aspirin")
    pairs = build_terminology_synonym_pairs(
        (
            ConceptEntry(
                **{
                    **metformin_500.__dict__,
                    "aliases": ("metformin 500",),
                }
            ),
            metformin_850,
            aspirin,
        )
    )

    examples = build_synonym_retrieval_examples(
        pairs,
        (metformin_500, metformin_850, aspirin),
        maximum_hard_negatives=2,
    )

    assert examples
    assert examples[0].hard_negatives[0] == "metformin 850 mg tablet"


class _Encoder:
    def encode(self, texts: tuple[str, ...]) -> list[tuple[float, ...]]:
        return [
            (0.0, 1.0) if text == "aspirin" else (1.0, 0.0)
            for text in texts
        ]


def _entry(
    concept_id: str,
    code: str,
    entity_type: EntityType,
    code_system: CodeSystem,
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> ConceptEntry:
    return ConceptEntry(
        concept_id=concept_id,
        code=code,
        code_system=code_system,
        canonical_name=name,
        semantic_type=entity_type,
        aliases=aliases,
    )


def _drug(
    concept_id: str,
    code: str,
    name: str,
    strength: str,
    *,
    ingredient: str = "metformin",
) -> ConceptEntry:
    return ConceptEntry(
        concept_id=concept_id,
        code=code,
        code_system=CodeSystem.RXNORM,
        canonical_name=name,
        semantic_type=EntityType.DRUG,
        ingredient=ingredient,
        strength=strength,
        dose_form="tablet",
    )
