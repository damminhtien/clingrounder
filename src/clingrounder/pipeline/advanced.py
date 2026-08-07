"""Advanced composition APIs for library and research integrations.

Application code should import :class:`clingrounder.Pipeline`.  This module is the single
public namespace for callers that need to inject ports, compose resources, or inspect runtime
stages directly.
"""

from __future__ import annotations

from clingrounder.pipeline.components import PipelineComponents
from clingrounder.pipeline.config_loader import ResolvedPipelineConfig
from clingrounder.pipeline.factory import PipelineFactory, PipelineConfig, TerminologyConfig
from clingrounder.pipeline.model_config import (
    ListwiseRerankerModelConfig,
    PipelineModelConfig,
)
from clingrounder.pipeline.options import PipelineOptions
from clingrounder.pipeline.parallel_batch import (
    ParallelBatchError,
    ParallelBatchOptions,
    PipelineBatchExecutor,
)
from clingrounder.pipeline.ports import (
    BatchCandidateRerankerPort,
    BatchCandidateRetrieverPort,
    CandidateRerankRequest,
    CandidateRetrievalRequest,
)
from clingrounder.pipeline.profile import (
    PIPELINE_PROFILE_SCHEMA_VERSION,
    PipelineProfileMetadata,
    ProfileMaturity,
)
from clingrounder.pipeline.runner import PipelineRunResult, PipelineRunner
from clingrounder.pipeline.subsystems import (
    ContextConfig,
    GraphEvidenceConfig,
    LinkingConfig,
    RelationsConfig,
    RuntimeConfig,
    ValidationConfig,
)
from clingrounder.pipeline.runtime import Closable, PipelineRuntime, RuntimeCapabilities
from clingrounder.pipeline.stages import (
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
from clingrounder.pipeline.tracing import (
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
    "ContextConfig",
    "DocumentPreparationStage",
    "DocumentStructure",
    "EntityKnowledgeValidationResult",
    "EntityKnowledgeValidationStage",
    "EntityExtractionStage",
    "GraphEvidenceRerankingStage",
    "GraphEvidenceConfig",
    "InMemoryPipelineObserver",
    "ListwiseRerankerModelConfig",
    "LinkingContext",
    "LinkingConfig",
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
    "PipelineConfig",
    "TerminologyConfig",
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
    "RelationsConfig",
    "RuntimeCapabilities",
    "RuntimeConfig",
    "StageMeasurement",
    "ValidationConfig",
]
