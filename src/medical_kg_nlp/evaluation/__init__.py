from __future__ import annotations
from medical_kg_nlp.evaluation.ablation import (
    AblationVariantResult,
    StageAggregate,
    aggregate_traces,
    flatten_metrics,
)
from medical_kg_nlp.evaluation.data_profile import profile_dataset, profile_paths, render_markdown
from medical_kg_nlp.evaluation.end_to_end_metrics import evaluate_predictions
from medical_kg_nlp.evaluation.loop_engineer import (
    build_loop_engineering_report,
    write_loop_engineering_report,
)
from medical_kg_nlp.evaluation.phase1 import (
    build_phase1_report,
    load_phase1_text_documents,
    prediction_to_phase1_entities,
    score_phase1_documents,
    validate_phase1_entities,
    validate_phase1_submission_dir,
    validate_phase1_submission_zip,
    write_phase1_output_dir,
    zip_phase1_output_dir,
)
from medical_kg_nlp.evaluation.pipeline_report import build_pipeline_report, write_pipeline_report

__all__ = [
    "AblationVariantResult",
    "StageAggregate",
    "aggregate_traces",
    "build_loop_engineering_report",
    "build_phase1_report",
    "build_pipeline_report",
    "evaluate_predictions",
    "flatten_metrics",
    "load_phase1_text_documents",
    "prediction_to_phase1_entities",
    "profile_dataset",
    "profile_paths",
    "render_markdown",
    "score_phase1_documents",
    "validate_phase1_entities",
    "validate_phase1_submission_dir",
    "validate_phase1_submission_zip",
    "write_loop_engineering_report",
    "write_phase1_output_dir",
    "write_pipeline_report",
    "zip_phase1_output_dir",
]
