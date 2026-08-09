"""Composable clinical NLP, terminology retrieval, and evaluation toolkit."""

from __future__ import annotations

from clingrounder.pipeline.facade import (
    Pipeline,
    PipelineClosedError,
    PipelineConfig,
    PipelineConfigurationError,
    UnknownProfileError,
    load_pipeline,
)
from clingrounder.schema.annotation import EntityAnnotation, RelationAnnotation
from clingrounder.schema.document import ClinicalDocument
from clingrounder.schema.output import ClinicalPrediction

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
    "load_pipeline",
    "__version__",
]

__version__ = "0.1.0a8"
