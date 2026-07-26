"""Phase 1 schema adapters, scorers, campaign tools, and manual-gold utilities."""

from medical_kg_nlp.benchmarks.phase1.adapter import (
    Phase1EvaluationAdapter,
    Phase1Record,
)
from medical_kg_nlp.benchmarks.phase1.phase1 import (
    Phase1SelectiveExportConfig,
    build_phase1_report,
    load_phase1_text_documents,
    prediction_to_phase1_entities,
    score_phase1_documents,
    validate_phase1_entities,
    validate_phase1_submission_documents,
    validate_phase1_submission_dir,
    validate_phase1_submission_zip,
    write_phase1_output_dir,
    zip_phase1_output_dir,
)
from medical_kg_nlp.benchmarks.phase1.manual_gold_mining import (
    Phase1ManualGoldMiningCorpus,
    build_phase1_reviewed_recognition_policy,
    load_phase1_manual_gold_mining_corpus,
)
from medical_kg_nlp.benchmarks.phase1.model_dataset import (
    PHASE1_FIVE_TYPE_LABELS,
    Phase1ModelDatasetConfig,
    build_phase1_model_dataset,
    build_phase1_model_splits,
)
from medical_kg_nlp.benchmarks.phase1.model_runtime import (
    run_phase1_model_calibration,
)
from medical_kg_nlp.benchmarks.phase1.model_selection import (
    PHASE1_NER_VARIANTS,
    Phase1HoldoutGate,
    Phase1ModelSelectionConfig,
    calibrate_phase1_model_thresholds,
    compare_phase1_ner_variants,
    infer_phase1_development_predictions,
)
from medical_kg_nlp.benchmarks.phase1.recognition_mining import (
    Phase1RecognitionMiningConfig,
    run_phase1_recognition_mining,
)
from medical_kg_nlp.benchmarks.phase1.round2 import (
    ROUND2_NOVELTY_SOURCE_IDS,
    build_phase1_round2_audit,
    load_phase1_round2_documents,
    write_phase1_round2_audit,
)
from medical_kg_nlp.benchmarks.phase1.round2_probes import (
    Phase1Round2ProbeConfig,
    Phase1TextRegion,
    RegionProposalPolicy,
    align_quoted_phase1_proposals,
    canonicalize_full_phase1_source,
    merge_region_routed_proposals,
    run_phase1_round2_probes,
    segment_phase1_text_regions,
)

__all__ = [
    "Phase1SelectiveExportConfig",
    "Phase1EvaluationAdapter",
    "Phase1Record",
    "Phase1ManualGoldMiningCorpus",
    "Phase1ModelDatasetConfig",
    "Phase1HoldoutGate",
    "Phase1ModelSelectionConfig",
    "Phase1RecognitionMiningConfig",
    "Phase1Round2ProbeConfig",
    "Phase1TextRegion",
    "RegionProposalPolicy",
    "PHASE1_FIVE_TYPE_LABELS",
    "PHASE1_NER_VARIANTS",
    "ROUND2_NOVELTY_SOURCE_IDS",
    "build_phase1_round2_audit",
    "build_phase1_reviewed_recognition_policy",
    "build_phase1_report",
    "build_phase1_model_dataset",
    "build_phase1_model_splits",
    "calibrate_phase1_model_thresholds",
    "canonicalize_full_phase1_source",
    "compare_phase1_ner_variants",
    "infer_phase1_development_predictions",
    "load_phase1_text_documents",
    "load_phase1_round2_documents",
    "merge_region_routed_proposals",
    "prediction_to_phase1_entities",
    "score_phase1_documents",
    "validate_phase1_entities",
    "validate_phase1_submission_documents",
    "validate_phase1_submission_dir",
    "validate_phase1_submission_zip",
    "write_phase1_output_dir",
    "zip_phase1_output_dir",
    "load_phase1_manual_gold_mining_corpus",
    "run_phase1_recognition_mining",
    "run_phase1_round2_probes",
    "run_phase1_model_calibration",
    "segment_phase1_text_regions",
    "align_quoted_phase1_proposals",
    "write_phase1_round2_audit",
]
