"""Framework-neutral training contracts and optional local model runtimes."""

from medical_kg_nlp.training.config import TokenClassifierTrainingConfig
from medical_kg_nlp.training.huggingface_token_classifier import (
    inspect_token_classifier_training_inputs,
    train_huggingface_token_classifier,
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
    TokenBoundaryAlignmentError,
    TokenizedTrainingWindow,
    compute_bio_span_metrics,
    decode_bio_spans,
    project_record_to_token_windows,
)

__all__ = [
    "FastTokenizerPort",
    "SpanDatasetSummary",
    "SpanTrainingEntity",
    "SpanTrainingRecord",
    "TokenBoundaryAlignmentError",
    "TokenClassifierTrainingConfig",
    "TokenizedTrainingWindow",
    "build_bio_label_vocabulary",
    "compute_bio_span_metrics",
    "decode_bio_spans",
    "inspect_token_classifier_training_inputs",
    "iter_span_training_records",
    "project_record_to_token_windows",
    "scan_span_dataset",
    "train_huggingface_token_classifier",
    "validate_span_dataset_manifest",
]
