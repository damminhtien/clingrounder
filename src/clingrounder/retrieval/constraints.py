"""Entity-type and code-system constraints shared by retriever adapters."""

from __future__ import annotations

from collections.abc import Sequence

from clingrounder.schema.types import CodeSystem, EntityType

__all__ = ["ALLOWED_CODE_SYSTEMS", "allowed_code_systems"]

ALLOWED_CODE_SYSTEMS: dict[EntityType, set[CodeSystem]] = {
    EntityType.DRUG: {CodeSystem.RXNORM},
    EntityType.DISEASE: {
        CodeSystem.ICD10,
        CodeSystem.MONDO,
        CodeSystem.UMLS,
        CodeSystem.SNOMED,
    },
    EntityType.SYMPTOM: {
        CodeSystem.HPO,
        CodeSystem.UMLS,
        CodeSystem.SNOMED,
        CodeSystem.LOCAL,
    },
    EntityType.LAB_TEST: {CodeSystem.LOCAL},
    EntityType.LAB_RESULT: {CodeSystem.NONE, CodeSystem.LOCAL},
    EntityType.DOSAGE: {CodeSystem.NONE},
    EntityType.STRENGTH: {CodeSystem.NONE},
    EntityType.FREQUENCY: {CodeSystem.NONE},
    EntityType.ROUTE: {CodeSystem.NONE},
    EntityType.DURATION: {CodeSystem.NONE},
    EntityType.DOSAGE_FORM: {CodeSystem.NONE},
    EntityType.PROCEDURE: {
        CodeSystem.ICD10,
        CodeSystem.UMLS,
        CodeSystem.SNOMED,
        CodeSystem.LOCAL,
    },
    EntityType.FINDING: {
        CodeSystem.HPO,
        CodeSystem.UMLS,
        CodeSystem.SNOMED,
        CodeSystem.LOCAL,
    },
}


def allowed_code_systems(entity_type: EntityType) -> Sequence[CodeSystem] | None:
    """Return a deterministic query filter for an entity type."""

    systems = ALLOWED_CODE_SYSTEMS.get(entity_type)
    return tuple(sorted(systems, key=lambda item: item.value)) if systems is not None else None
