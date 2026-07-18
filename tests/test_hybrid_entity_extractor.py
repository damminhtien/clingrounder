"""Hybrid NER tests for reviewed-span precedence and raw offset safety."""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.adapters.hybrid import HybridEntityExtractorAdapter
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match


@dataclass(frozen=True)
class _Extractor:
    entities: list[EntityAnnotation]

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        return self.entities


def _entity(
    entity_id: str,
    source: str,
    start: int,
    end: int,
    entity_type: EntityType,
    confidence: float,
) -> EntityAnnotation:
    text = source[start:end]
    return EntityAnnotation(
        id=entity_id,
        span=(start, end),
        text=text,
        normalized_text=normalize_for_match(text),
        type=entity_type,
        confidence=confidence,
    )


def test_hybrid_dictionary_wins_overlap_and_model_adds_recall() -> None:
    source = "bệnh đau ngực và sốt"
    fever_start = source.index("sốt")
    dictionary = _Extractor(
        [_entity("D1", source, 0, 13, EntityType.DISEASE, 0.8)]
    )
    model = _Extractor(
        [
            _entity("M1", source, 5, 13, EntityType.SYMPTOM, 0.99),
            _entity("M2", source, fever_start, len(source), EntityType.SYMPTOM, 0.7),
        ]
    )

    entities = HybridEntityExtractorAdapter(model=model, dictionary=dictionary).extract(source)

    assert [(entity.text, entity.type) for entity in entities] == [
        ("bệnh đau ngực", EntityType.DISEASE),
        ("sốt", EntityType.SYMPTOM),
    ]
    assert [entity.id for entity in entities] == ["H0001", "H0002"]
    for entity in entities:
        entity.validate_offsets(source)


def test_hybrid_deterministically_resolves_overlapping_model_spans() -> None:
    source = "đau ngực dữ dội"
    model = _Extractor(
        [
            _entity("short", source, 0, 8, EntityType.SYMPTOM, 0.8),
            _entity("long", source, 0, len(source), EntityType.SYMPTOM, 0.8),
        ]
    )

    entities = HybridEntityExtractorAdapter(
        model=model,
        dictionary=_Extractor([]),
    ).extract(source)

    assert [(entity.text, entity.span) for entity in entities] == [
        (source, (0, len(source)))
    ]
