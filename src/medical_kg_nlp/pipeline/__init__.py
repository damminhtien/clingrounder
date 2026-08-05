"""Pipeline contracts, composition, execution, and parallel batch APIs."""

from __future__ import annotations

from medical_kg_nlp.pipeline.components import PipelineComponents
from medical_kg_nlp.pipeline.config_loader import ResolvedPipelineConfig
from medical_kg_nlp.pipeline.factory import PipelineFactory, PipelineFactoryConfig
from medical_kg_nlp.pipeline.model_config import (
    ListwiseRerankerModelConfig,
    PipelineModelConfig,
)
from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.pipeline.ports import (
    BatchCandidateRerankerPort,
    BatchCandidateRetrieverPort,
    CandidateRerankRequest,
    CandidateRetrievalRequest,
)
from medical_kg_nlp.pipeline.parallel_batch import (
    ParallelBatchError,
    ParallelBatchOptions,
    PipelineBatchExecutor,
)
from medical_kg_nlp.pipeline.profile import (
    PIPELINE_PROFILE_SCHEMA_VERSION,
    PipelineProfileMetadata,
    ProfileMaturity,
)
from medical_kg_nlp.pipeline.runner import PipelineRunResult, PipelineRunner
from medical_kg_nlp.pipeline.runtime import Closable, PipelineRuntime, RuntimeCapabilities
from medical_kg_nlp.pipeline.tracing import (
    InMemoryPipelineObserver,
    NoOpPipelineObserver,
    OpenTelemetryPipelineObserver,
    PipelineObserverPort,
    PipelineTrace,
    StageMeasurement,
)

__all__ = [
    "ParallelBatchError",
    "ParallelBatchOptions",
    "PipelineBatchExecutor",
    "PIPELINE_PROFILE_SCHEMA_VERSION",
    "PipelineComponents",
    "PipelineFactory",
    "PipelineFactoryConfig",
    "ListwiseRerankerModelConfig",
    "PipelineModelConfig",
    "PipelineOptions",
    "PipelineRuntime",
    "Closable",
    "BatchCandidateRerankerPort",
    "BatchCandidateRetrieverPort",
    "CandidateRerankRequest",
    "CandidateRetrievalRequest",
    "PipelineProfileMetadata",
    "ProfileMaturity",
    "ResolvedPipelineConfig",
    "PipelineRunResult",
    "PipelineRunner",
    "RuntimeCapabilities",
    "PipelineTrace",
    "StageMeasurement",
    "PipelineObserverPort",
    "InMemoryPipelineObserver",
    "NoOpPipelineObserver",
    "OpenTelemetryPipelineObserver",
]
