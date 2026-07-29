"""Task-neutral adapters, metrics, error analysis, and report rendering."""

from medical_kg_nlp.evaluation.adapters import EvaluationAdapter, adapt_evaluation_records
from medical_kg_nlp.evaluation.context_metrics import assertion_attribute_metrics
from medical_kg_nlp.evaluation.data_profile import profile_dataset, profile_paths, render_markdown
from medical_kg_nlp.evaluation.end_to_end_metrics import evaluate_predictions
from medical_kg_nlp.evaluation.pipeline_report import build_pipeline_report, write_pipeline_report
from medical_kg_nlp.evaluation.records import (
    EvaluationDocument,
    EvaluationEntity,
    EvaluationRelation,
)
from medical_kg_nlp.evaluation.sparse_logistic import (
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
    "binary_probability_metrics",
    "build_pipeline_report",
    "evaluate_predictions",
    "fit_sparse_logistic",
    "profile_dataset",
    "profile_paths",
    "render_markdown",
    "write_pipeline_report",
]
