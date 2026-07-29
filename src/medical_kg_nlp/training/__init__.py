"""Framework-neutral training contracts and optional local model runtimes."""

from medical_kg_nlp.training.causal_artifact import finalize_causal_qlora_artifact
from medical_kg_nlp.training.causal_instruction import (
    CausalInstructionRecord,
    CausalInstructionSource,
    InstructionDatasetReport,
    InstructionTooLongError,
    load_causal_instruction_records,
    tokenize_causal_instruction,
)
from medical_kg_nlp.training.causal_qlora import (
    inspect_causal_qlora_inputs,
    train_causal_qlora,
)
from medical_kg_nlp.training.causal_run_spec import (
    CausalQLoRAConfig,
    CausalQLoRARunSpec,
    load_causal_qlora_run_spec,
)
from medical_kg_nlp.training.config import TokenClassifierTrainingConfig
from medical_kg_nlp.training.huggingface_token_classifier import (
    fingerprint_model_directory,
    inspect_token_classifier_training_inputs,
    train_huggingface_token_classifier,
    verify_token_classifier_artifact,
    verify_saved_token_classifier,
)
from medical_kg_nlp.training.run_spec import (
    GPURequirements,
    TokenClassifierRunSpec,
    assert_local_gpu_runtime,
    inspect_local_runtime,
    load_token_classifier_run_spec,
    verify_token_classifier_run_artifact,
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
from medical_kg_nlp.training.terminology_pairs import (
    SynonymPairMode,
    TerminologyPairConfig,
    TerminologySynonymPair,
    build_terminology_synonym_pairs,
    write_terminology_pair_dataset,
)

__all__ = [
    "CausalInstructionRecord",
    "CausalInstructionSource",
    "CausalQLoRAConfig",
    "CausalQLoRARunSpec",
    "FastTokenizerPort",
    "GPURequirements",
    "InstructionDatasetReport",
    "InstructionTooLongError",
    "SpanDatasetSummary",
    "SpanTrainingEntity",
    "SpanTrainingRecord",
    "SynonymPairMode",
    "TerminologyPairConfig",
    "TerminologySynonymPair",
    "TokenAlignmentPolicy",
    "TokenBoundaryAlignmentError",
    "TokenClassifierTrainingConfig",
    "TokenClassifierRunSpec",
    "TokenizedTrainingWindow",
    "build_bio_label_vocabulary",
    "build_terminology_synonym_pairs",
    "assert_local_gpu_runtime",
    "compute_bio_span_metrics",
    "decode_bio_spans",
    "finalize_causal_qlora_artifact",
    "fingerprint_model_directory",
    "find_unaligned_annotations",
    "inspect_token_classifier_training_inputs",
    "inspect_local_runtime",
    "inspect_causal_qlora_inputs",
    "iter_span_training_records",
    "load_causal_instruction_records",
    "load_causal_qlora_run_spec",
    "project_record_to_token_windows",
    "load_token_classifier_run_spec",
    "scan_span_dataset",
    "train_huggingface_token_classifier",
    "tokenize_causal_instruction",
    "train_causal_qlora",
    "validate_span_dataset_manifest",
    "verify_token_classifier_artifact",
    "verify_token_classifier_run_artifact",
    "verify_saved_token_classifier",
    "write_terminology_pair_dataset",
]
