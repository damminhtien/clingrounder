"""Advanced composition APIs for library and research integrations.

Application code should import :class:`medical_kg_nlp.Pipeline`.  This module is the single
public namespace for callers that need to inject ports, compose resources, or inspect runtime
stages directly.
"""

from __future__ import annotations

from medical_kg_nlp.pipeline.components import PipelineComponents
from medical_kg_nlp.pipeline.config_loader import ResolvedPipelineConfig
from medical_kg_nlp.pipeline.factory import PipelineFactory, PipelineFactoryConfig
from medical_kg_nlp.pipeline.model_config import (
    ListwiseRerankerModelConfig,
    PipelineModelConfig,
)
from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.pipeline.parallel_batch import (
    ParallelBatchError,
    ParallelBatchOptions,
    PipelineBatchExecutor,
)
from medical_kg_nlp.pipeline.ports import (
    BatchCandidateRerankerPort,
    BatchCandidateRetrieverPort,
    CandidateRerankRequest,
    CandidateRetrievalRequest,
)
from medical_kg_nlp.pipeline.profile import (
    PIPELINE_PROFILE_SCHEMA_VERSION,
    PipelineProfileMetadata,
    ProfileMaturity,
)
from medical_kg_nlp.pipeline.runner import PipelineRunResult, PipelineRunner
from medical_kg_nlp.pipeline.runtime import Closable, PipelineRuntime, RuntimeCapabilities
from medical_kg_nlp.pipeline.stages import (
    AssertionClassificationStage,
    CandidateGenerationResult,
    CandidateGenerationStage,
    CandidateRerankingResult,
    CandidateRerankingStage,
    DocumentPreparationStage,
    DocumentStructure,
    EntityKnowledgeValidationResult,
    EntityKnowledgeValidationStage,
    EntityExtractionStage,
    GraphEvidenceRerankingStage,
    LinkingContext,
    LinkingStageResult,
    NormalizationAssignmentStage,
    PredictionValidationResult,
    PredictionValidationStage,
    PreparedDocument,
    RelationExtractionResult,
    RelationExtractionStage,
)
from medical_kg_nlp.pipeline.tracing import (
    InMemoryPipelineObserver,
    NoOpPipelineObserver,
    OpenTelemetryPipelineObserver,
    PipelineObserverPort,
    PipelineTrace,
    StageMeasurement,
)

__all__ = [
    "BatchCandidateRerankerPort",
    "BatchCandidateRetrieverPort",
    "AssertionClassificationStage",
    "CandidateGenerationResult",
    "CandidateGenerationStage",
    "CandidateRerankingResult",
    "CandidateRerankingStage",
    "CandidateRerankRequest",
    "CandidateRetrievalRequest",
    "Closable",
    "DocumentPreparationStage",
    "DocumentStructure",
    "EntityKnowledgeValidationResult",
    "EntityKnowledgeValidationStage",
    "EntityExtractionStage",
    "GraphEvidenceRerankingStage",
    "InMemoryPipelineObserver",
    "ListwiseRerankerModelConfig",
    "LinkingContext",
    "LinkingStageResult",
    "NoOpPipelineObserver",
    "NormalizationAssignmentStage",
    "OpenTelemetryPipelineObserver",
    "PIPELINE_PROFILE_SCHEMA_VERSION",
    "ParallelBatchError",
    "ParallelBatchOptions",
    "PipelineBatchExecutor",
    "PipelineComponents",
    "PipelineFactory",
    "PipelineFactoryConfig",
    "PipelineModelConfig",
    "PipelineObserverPort",
    "PipelineOptions",
    "PipelineProfileMetadata",
    "PipelineRunResult",
    "PipelineRunner",
    "PipelineRuntime",
    "PipelineTrace",
    "PredictionValidationResult",
    "PredictionValidationStage",
    "PreparedDocument",
    "ProfileMaturity",
    "ResolvedPipelineConfig",
    "RelationExtractionResult",
    "RelationExtractionStage",
    "RuntimeCapabilities",
    "StageMeasurement",
]
