"""Typed internal document, annotation, prediction, and validator schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from medical_kg_nlp.schema.annotation import (
    AmbiguousEntityProposal,
    CandidateConcept,
    EntityAnnotation,
    EntityExtractionResult,
    RelationAnnotation,
)
from medical_kg_nlp.schema.document import ClinicalDocument, Section, Sentence
from medical_kg_nlp.schema.output import ClinicalPrediction, PredictionMetadata
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType, RelationType

if TYPE_CHECKING:
    from medical_kg_nlp.schema.validator import (
        PredictionValidationIssue,
        PredictionValidator,
        prediction_from_json,
    )

__all__ = [
    "AssertionStatus",
    "AmbiguousEntityProposal",
    "CandidateConcept",
    "ClinicalDocument",
    "ClinicalPrediction",
    "CodeSystem",
    "EntityAnnotation",
    "EntityExtractionResult",
    "EntityType",
    "PredictionMetadata",
    "PredictionValidationIssue",
    "PredictionValidator",
    "RelationAnnotation",
    "RelationType",
    "Section",
    "Sentence",
    "prediction_from_json",
]


def __getattr__(name: str) -> Any:
    if name in {"PredictionValidationIssue", "PredictionValidator", "prediction_from_json"}:
        from medical_kg_nlp.schema import validator

        return getattr(validator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
