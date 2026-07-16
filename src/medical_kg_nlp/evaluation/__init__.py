"""Task-neutral adapters, metrics, error analysis, and report rendering."""

from medical_kg_nlp.evaluation.adapters import EvaluationAdapter, adapt_evaluation_records
from medical_kg_nlp.evaluation.data_profile import profile_dataset, profile_paths, render_markdown
from medical_kg_nlp.evaluation.end_to_end_metrics import evaluate_predictions
from medical_kg_nlp.evaluation.pipeline_report import build_pipeline_report, write_pipeline_report
from medical_kg_nlp.evaluation.records import (
    EvaluationDocument,
    EvaluationEntity,
    EvaluationRelation,
)

__all__ = [
    "EvaluationAdapter",
    "EvaluationDocument",
    "EvaluationEntity",
    "EvaluationRelation",
    "adapt_evaluation_records",
    "build_pipeline_report",
    "evaluate_predictions",
    "profile_dataset",
    "profile_paths",
    "render_markdown",
    "write_pipeline_report",
]
