"""Source-label harmonization tests for terminology-safe mined annotations."""

from __future__ import annotations

import json

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.dictionaries.synonym_table import ConceptEntry
from clingrounder.mining.harmonization import (
    AnnotationHarmonizationPolicy,
    AnnotationHarmonizationRule,
    harmonize_annotations,
)
from clingrounder.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)
from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.terminology.memory import InMemoryTerminologyRepository


def _document() -> MinedDocument:
    return MinedDocument(
        document_id="codiesp:doc",
        text="hipertensión y código extendido",
        language="es",
        note_type="clinical_case",
        source_artifact_id="codiesp:artifact",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        metadata={"source_id": "codiesp"},
    )


def _annotation(
    annotation_id: str, span: tuple[int, int], code: str
) -> AnnotationProposal:
    document = _document()
    return AnnotationProposal(
        annotation_id=annotation_id,
        document_id=document.document_id,
        span=span,
        text=document.text[span[0] : span[1]],
        entity_type="FINDING",
        assertions=(),
        concepts=(
            ConceptLink(
                code_system="ICD-10-CM",
                code=code,
                terminology_version="codiesp-source",
            ),
        ),
        confidence=1.0,
        layer=AnnotationLayer.SILVER,
        label_source="source_human_annotation",
        labeler_id="codiesp-source",
        review_status=ReviewStatus.PROPOSED,
        source_label="DIAGNOSTICO",
    )


def test_harmonization_maps_known_code_and_drops_unsupported_extension() -> None:
    document = _document()
    known = _annotation("known", (0, 11), "I10")
    unknown = _annotation("unknown", (14, len(document.text)), "I10.EXT")
    repository = InMemoryTerminologyRepository(
        DictionaryStore(
            [
                ConceptEntry(
                    concept_id="icd10:I10",
                    code="I10",
                    code_system=CodeSystem.ICD10,
                    canonical_name="Tăng huyết áp",
                    semantic_type=EntityType.DISEASE,
                )
            ]
        )
    )
    policy = AnnotationHarmonizationPolicy(
        schema_version="annotation-harmonization-policy.v1",
        policy_id="fixture",
        rules=(
            AnnotationHarmonizationRule(
                rule_id="codiesp-diagnosis",
                source_id="codiesp",
                source_entity_type="FINDING",
                source_label="DIAGNOSTICO",
                target_entity_type=EntityType.DISEASE,
                concept_system_map={"ICD-10-CM": CodeSystem.ICD10},
                target_terminology_version="TT06-fixture",
                unmapped_concept_action="drop",
            ),
        ),
    )

    result = harmonize_annotations(
        (document,), (known, unknown), repository, policy
    )
    by_id = {annotation.annotation_id: annotation for annotation in result.annotations}

    assert by_id["known"].entity_type == "DISEASE"
    assert by_id["known"].concepts[0].code_system == "ICD-10"
    assert by_id["known"].concepts[0].terminology_version == "TT06-fixture"
    assert by_id["unknown"].concepts == ()
    dropped = json.loads(by_id["unknown"].metadata["harmonization_unmapped_concepts"])
    assert dropped[0]["code"] == "I10.EXT"
    assert result.report["decision_counts"] == {
        "harmonized_annotations": 2,
        "mapped_concepts": 1,
        "unmapped_concepts": 1,
    }
    for annotation in result.annotations:
        annotation.validate_offsets(document)
