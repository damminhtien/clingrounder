from __future__ import annotations
from medical_kg_nlp.evaluation.ablation import (
    AblationVariantResult,
    StageAggregate,
    aggregate_traces,
    flatten_metrics,
)
from medical_kg_nlp.evaluation.annotation_knowledge import (
    compile_annotation_knowledge,
    render_annotation_knowledge_markdown,
    write_annotation_knowledge,
)
from medical_kg_nlp.evaluation.data_profile import profile_dataset, profile_paths, render_markdown
from medical_kg_nlp.evaluation.end_to_end_metrics import evaluate_predictions
from medical_kg_nlp.evaluation.entity_wer_report import (
    build_entity_wer_report,
    render_entity_wer_markdown,
    write_entity_wer_report,
)
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
from medical_kg_nlp.evaluation.phase1_candidate_overlay import (
    Phase1CandidateIndex,
    Phase1CandidateOverlayConfig,
    apply_phase1_candidate_overlay,
    candidate_ablation_passes,
)
from medical_kg_nlp.evaluation.phase1_ensemble import (
    expand_repeated_phase1_mentions,
    load_phase1_output_source,
    merge_phase1_outputs,
    rank_phase1_source_strategies,
)
from medical_kg_nlp.evaluation.phase1_submission_analysis import (
    build_phase1_submission_analysis,
    render_phase1_submission_analysis,
    write_phase1_submission_analysis,
)
from medical_kg_nlp.evaluation.pipeline_report import build_pipeline_report, write_pipeline_report

__all__ = [
    "AblationVariantResult",
    "Phase1CandidateIndex",
    "Phase1CandidateOverlayConfig",
    "StageAggregate",
    "aggregate_traces",
    "apply_phase1_candidate_overlay",
    "build_loop_engineering_report",
    "build_phase1_report",
    "build_phase1_submission_analysis",
    "build_pipeline_report",
    "build_entity_wer_report",
    "candidate_ablation_passes",
    "compile_annotation_knowledge",
    "evaluate_predictions",
    "expand_repeated_phase1_mentions",
    "flatten_metrics",
    "load_phase1_text_documents",
    "load_phase1_output_source",
    "merge_phase1_outputs",
    "prediction_to_phase1_entities",
    "profile_dataset",
    "profile_paths",
    "rank_phase1_source_strategies",
    "render_markdown",
    "render_annotation_knowledge_markdown",
    "render_entity_wer_markdown",
    "render_phase1_submission_analysis",
    "score_phase1_documents",
    "validate_phase1_entities",
    "validate_phase1_submission_dir",
    "validate_phase1_submission_zip",
    "write_loop_engineering_report",
    "write_annotation_knowledge",
    "write_entity_wer_report",
    "write_phase1_output_dir",
    "write_phase1_submission_analysis",
    "write_pipeline_report",
    "zip_phase1_output_dir",
]
