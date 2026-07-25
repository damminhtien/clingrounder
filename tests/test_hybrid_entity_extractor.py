"""Hybrid NER tests for evidence arbitration and raw offset safety."""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.adapters.hybrid import (
    HybridArbitrationPolicy,
    HybridEntityExtractorAdapter,
)
from medical_kg_nlp.schema.annotation import (
    AmbiguousEntityProposal,
    EntityAnnotation,
    EntityExtractionResult,
)
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match


@dataclass(frozen=True)
class _Extractor:
    entities: list[EntityAnnotation]

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        return self.entities


@dataclass(frozen=True)
class _ProposalExtractor:
    result: EntityExtractionResult

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        return list(self.result.entities)

    def extract_with_proposals(self, source_text: str) -> EntityExtractionResult:
        return self.result


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


def test_hybrid_stronger_model_can_correct_overlapping_dictionary_span() -> None:
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
        ("đau ngực", EntityType.SYMPTOM),
        ("sốt", EntityType.SYMPTOM),
    ]
    assert [entity.id for entity in entities] == ["H0001", "H0002"]
    for entity in entities:
        entity.validate_offsets(source)


def test_hybrid_dictionary_prior_breaks_small_confidence_gap() -> None:
    source = "bệnh đau ngực"
    dictionary = _Extractor(
        [_entity("D1", source, 0, len(source), EntityType.DISEASE, 0.78)]
    )
    model = _Extractor(
        [_entity("M1", source, 5, len(source), EntityType.SYMPTOM, 0.80)]
    )

    entities = HybridEntityExtractorAdapter(model=model, dictionary=dictionary).extract(source)

    assert [(entity.text, entity.type) for entity in entities] == [
        (source, EntityType.DISEASE)
    ]


def test_hybrid_exact_agreement_merges_duplicate_and_confidence() -> None:
    source = "đau ngực"
    dictionary = _Extractor(
        [_entity("D1", source, 0, len(source), EntityType.SYMPTOM, 0.78)]
    )
    model = _Extractor(
        [_entity("M1", source, 0, len(source), EntityType.SYMPTOM, 0.93)]
    )

    entities = HybridEntityExtractorAdapter(model=model, dictionary=dictionary).extract(source)

    assert len(entities) == 1
    assert entities[0].confidence == 0.93
    assert entities[0].id == "H0001"


def test_hybrid_maximizes_evidence_across_overlap_component() -> None:
    source = "đau ngực và sốt"
    fever_start = source.index("sốt")
    dictionary = _Extractor(
        [_entity("D1", source, 0, len(source), EntityType.DISEASE, 0.90)]
    )
    model = _Extractor(
        [
            _entity("M1", source, 0, 8, EntityType.SYMPTOM, 0.70),
            _entity("M2", source, fever_start, len(source), EntityType.SYMPTOM, 0.70),
        ]
    )

    entities = HybridEntityExtractorAdapter(model=model, dictionary=dictionary).extract(source)

    assert [entity.text for entity in entities] == ["đau ngực", "sốt"]


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


def test_hybrid_uses_ambiguous_dictionary_proposal_only_as_model_support() -> None:
    source = "khái niệm mơ hồ"
    model = _Extractor(
        [_entity("M1", source, 0, len(source), EntityType.DRUG, 0.70)]
    )
    dictionary = _ProposalExtractor(
        EntityExtractionResult(
            entities=(),
            ambiguous_proposals=(
                AmbiguousEntityProposal(
                    span=(0, len(source)),
                    text=source,
                    normalized_text=normalize_for_match(source),
                    candidate_types=(EntityType.DISEASE, EntityType.DRUG),
                    concept_ids=("D:1", "RX:1"),
                    confidence=0.78,
                ),
            ),
        )
    )

    entities = HybridEntityExtractorAdapter(
        model=model,
        dictionary=dictionary,
    ).extract(source)

    assert [(entity.text, entity.type, entity.confidence) for entity in entities] == [
        (source, EntityType.DRUG, 0.74)
    ]


def test_hybrid_policy_rejects_invalid_priors() -> None:
    for value in (-0.1, 1.1, float("nan")):
        try:
            HybridArbitrationPolicy(dictionary_prior=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid prior {value!r}")
