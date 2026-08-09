"""Task-neutral adapters, metrics, error analysis, and report rendering."""

from clingrounder.evaluation.adapters import EvaluationAdapter, adapt_evaluation_records
from clingrounder.evaluation.context_metrics import assertion_attribute_metrics
from clingrounder.evaluation.dataset_audit import DatasetAuditReport, audit_dataset
from clingrounder.evaluation.data_profile import profile_dataset, profile_paths, render_markdown
from clingrounder.evaluation.end_to_end_metrics import evaluate_predictions
from clingrounder.evaluation.pipeline_report import build_pipeline_report, write_pipeline_report
from clingrounder.evaluation.records import (
    EvaluationDocument,
    EvaluationEntity,
    EvaluationRelation,
)
from clingrounder.evaluation.relation_slices import relation_slice_counts
from clingrounder.evaluation.review_agreement import ReviewAgreementArtifact
from clingrounder.evaluation.linking_batch_benchmark import (
    CandidateBatchBenchmarkReport,
    benchmark_candidate_reranker,
)
from clingrounder.evaluation.sparse_logistic import (
    SparseBinaryExample,
    SparseLogisticModel,
    SparseLogisticTrainingConfig,
    binary_probability_metrics,
    fit_sparse_logistic,
)

__all__ = [
    "EvaluationAdapter",
    "EvaluationDocument",
    "EvaluationEntity",
    "EvaluationRelation",
    "SparseBinaryExample",
    "SparseLogisticModel",
    "SparseLogisticTrainingConfig",
    "adapt_evaluation_records",
    "assertion_attribute_metrics",
    "audit_dataset",
    "binary_probability_metrics",
    "build_pipeline_report",
    "CandidateBatchBenchmarkReport",
    "DatasetAuditReport",
    "evaluate_predictions",
    "fit_sparse_logistic",
    "profile_dataset",
    "profile_paths",
    "relation_slice_counts",
    "ReviewAgreementArtifact",
    "benchmark_candidate_reranker",
    "render_markdown",
    "write_pipeline_report",
]
