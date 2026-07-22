"""Framework-neutral training contracts and optional local model runtimes."""

from medical_kg_nlp.training.config import TokenClassifierTrainingConfig
from medical_kg_nlp.training.huggingface_token_classifier import (
    inspect_token_classifier_training_inputs,
    train_huggingface_token_classifier,
    verify_saved_token_classifier,
)
from medical_kg_nlp.training.run_spec import (
    GPURequirements,
    TokenClassifierRunSpec,
    assert_local_gpu_runtime,
    inspect_local_runtime,
    load_token_classifier_run_spec,
)
from medical_kg_nlp.training.span_dataset import (
    SpanDatasetSummary,
    SpanTrainingEntity,
    SpanTrainingRecord,
    build_bio_label_vocabulary,
    iter_span_training_records,
    scan_span_dataset,
    validate_span_dataset_manifest,
)
from medical_kg_nlp.training.token_labels import (
    FastTokenizerPort,
    TokenAlignmentPolicy,
    TokenBoundaryAlignmentError,
    TokenizedTrainingWindow,
    compute_bio_span_metrics,
    decode_bio_spans,
    find_unaligned_annotations,
    project_record_to_token_windows,
)

__all__ = [
    "FastTokenizerPort",
    "GPURequirements",
    "SpanDatasetSummary",
    "SpanTrainingEntity",
    "SpanTrainingRecord",
    "TokenAlignmentPolicy",
    "TokenBoundaryAlignmentError",
    "TokenClassifierTrainingConfig",
    "TokenClassifierRunSpec",
    "TokenizedTrainingWindow",
    "build_bio_label_vocabulary",
    "assert_local_gpu_runtime",
    "compute_bio_span_metrics",
    "decode_bio_spans",
    "find_unaligned_annotations",
    "inspect_token_classifier_training_inputs",
    "inspect_local_runtime",
    "iter_span_training_records",
    "project_record_to_token_windows",
    "load_token_classifier_run_spec",
    "scan_span_dataset",
    "train_huggingface_token_classifier",
    "validate_span_dataset_manifest",
    "verify_saved_token_classifier",
]
