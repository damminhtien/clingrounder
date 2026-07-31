"""Materialize a pinned five-type token model as joint-span proposal evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.adapters.huggingface import (
    HuggingFaceModelConfig,
    HuggingFaceTokenClassifierAdapter,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_sources import (
    EntityProposalExtractorPort,
    build_phase1_token_model_proposal_rows,
    write_phase1_joint_span_source_artifact,
)
from medical_kg_nlp.benchmarks.phase1.model_dataset import PHASE1_FIVE_TYPE_LABELS
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.hashing import sha256_directory

__all__ = [
    "PHASE1_TOKEN_LABEL_MAP",
    "Phase1TokenSourceConfig",
    "materialize_phase1_token_model_source",
]


PHASE1_TOKEN_LABEL_MAP = {
    entity_type.value: entity_type
    for entity_type in (
        EntityType.SYMPTOM,
        EntityType.LAB_TEST,
        EntityType.LAB_RESULT,
        EntityType.DISEASE,
        EntityType.DRUG,
    )
}


@dataclass(frozen=True, slots=True)
class Phase1TokenSourceConfig:
    """Pinned local token-model settings for a reusable Phase 1 proposal source."""

    model_path: Path
    model_fingerprint: str
    model_id: str
    base_revision: str
    device: str = "cpu"
    batch_size: int = 16
    max_length: int = 512
    stride: int = 64
    default_confidence_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not self.model_fingerprint.strip():
            raise ValueError("Token source model fingerprint must be non-empty")
        if not self.model_id.strip() or not self.base_revision.strip():
            raise ValueError("Token source model identity must be pinned")
        if self.stride < 0 or self.stride >= self.max_length - 2:
            raise ValueError("Token source stride must fit inside max_length")
        if not 0.0 <= self.default_confidence_threshold <= 1.0:
            raise ValueError("Token source confidence threshold must be within [0, 1]")


def materialize_phase1_token_model_source(
    corpus: Phase1ReviewedCorpus,
    config: Phase1TokenSourceConfig,
    *,
    output_dir: str | Path,
    source_name: str = "xlmr",
    extractor: EntityProposalExtractorPort | None = None,
) -> dict[str, Any]:
    """Project a local XLM-R checkpoint into an immutable all-document source artifact.

    ``extractor`` is injectable for offline contract tests. Production calls construct the local
    Hugging Face adapter from the checkpoint directory after verifying its content fingerprint.

    MODEL: ``model_path`` is the exact final token-classifier checkpoint. ``base_revision`` records
    the originating encoder revision separately, so a derived local directory remains traceable.
    """

    if not config.model_path.is_dir():
        raise FileNotFoundError(config.model_path)
    observed_fingerprint = sha256_directory(config.model_path)
    if observed_fingerprint != config.model_fingerprint:
        raise ValueError(
            "Token source checkpoint fingerprint mismatch: "
            f"expected={config.model_fingerprint}, observed={observed_fingerprint}"
        )
    active_extractor = extractor or HuggingFaceTokenClassifierAdapter(
        HuggingFaceModelConfig(
            model_id=str(config.model_path),
            # ``revision`` remains mandatory for adapter provenance. Local loading is restricted
            # to this directory, so transformers never resolves this fingerprint remotely.
            revision=config.model_fingerprint,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
        ),
        label_map=PHASE1_TOKEN_LABEL_MAP,
        stride=config.stride,
        default_confidence_threshold=config.default_confidence_threshold,
    )
    rows_by_document = build_phase1_token_model_proposal_rows(
        corpus,
        active_extractor,
        source_name=source_name,
    )
    return write_phase1_joint_span_source_artifact(
        corpus,
        rows_by_document,
        output_dir=output_dir,
        source_name=source_name,
        provenance={
            "kind": "local_token_classifier",
            "model_id": config.model_id,
            "base_revision": config.base_revision,
            "checkpoint_path": str(config.model_path),
            "checkpoint_sha256": config.model_fingerprint,
            "label_map": {
                label: entity_type.value for label, entity_type in PHASE1_TOKEN_LABEL_MAP.items()
            },
            "labels": list(PHASE1_FIVE_TYPE_LABELS),
            "device": config.device,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "stride": config.stride,
            "default_confidence_threshold": config.default_confidence_threshold,
        },
    )
