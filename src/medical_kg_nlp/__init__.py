"""Composable clinical NLP, terminology retrieval, and evaluation toolkit."""

from __future__ import annotations

from medical_kg_nlp.pipeline.facade import (
    Pipeline,
    PipelineClosedError,
    PipelineConfig,
    PipelineConfigurationError,
    UnknownProfileError,
)
from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction

__all__ = [
    "ClinicalDocument",
    "ClinicalPrediction",
    "EntityAnnotation",
    "Pipeline",
    "PipelineClosedError",
    "PipelineConfig",
    "PipelineConfigurationError",
    "RelationAnnotation",
    "UnknownProfileError",
    "__version__",
]

__version__ = "0.2.0"
