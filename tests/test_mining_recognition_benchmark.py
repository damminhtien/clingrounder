"""Recognition benchmarks gate mined lexicons on exact raw spans and types."""

from __future__ import annotations

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.mining.recognition_benchmark import benchmark_recognition_dictionary
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_mined_recognition_dictionary_improves_exact_drug_recall() -> None:
    text = "Given Drug One Hundred today."
    start = text.index("Drug One Hundred")
    document = MinedDocument(
        document_id="doc-1",
        text=text,
        language="en",
        note_type="medication_record",
        source_artifact_id="artifact-1",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
    )
    gold = AnnotationProposal(
        annotation_id="gold-1",
        document_id=document.document_id,
        span=(start, start + len("Drug One Hundred")),
        text="Drug One Hundred",
        entity_type="DRUG",
        assertions=(),
        concepts=(),
        confidence=1.0,
        layer=AnnotationLayer.GOLD,
        label_source="fixture",
        labeler_id="fixture",
        review_status=ReviewStatus.ACCEPTED,
    )
    baseline = DictionaryStore([_concept("RX:1", "1", "metformin", ())])
    additional = DictionaryStore(
        [_concept("RX:100", "100", "Drug 100", ("Drug One Hundred",))]
    )

    report = benchmark_recognition_dictionary(
        (document,),
        (gold,),
        baseline,
        additional,
        entity_types=(EntityType.DRUG,),
    )

    assert report["baseline"]["metrics"]["recall"] == 0.0
    assert report["enriched"]["metrics"]["recall"] == 1.0
    assert report["delta"]["true_positive_count"] == 1
    assert report["enriched"]["runtime_ms"]["per_document_max"] >= 0.0


def _concept(
    concept_id: str,
    code: str,
    name: str,
    aliases: tuple[str, ...],
) -> ConceptEntry:
    return ConceptEntry(
        concept_id=concept_id,
        code=code,
        code_system=CodeSystem.RXNORM,
        canonical_name=name,
        semantic_type=EntityType.DRUG,
        aliases=aliases,
        source="fixture",
    )
