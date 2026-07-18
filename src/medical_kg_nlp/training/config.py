"""Pinned configuration for reproducible local token-classifier training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["TokenClassifierTrainingConfig"]


@dataclass(frozen=True)
class TokenClassifierTrainingConfig:
    """Dataset, model identity, and bounded training hyperparameters."""

    dataset_path: Path
    dataset_manifest_path: Path
    output_dir: Path
    model_id: str
    revision: str
    train_split: str = "train"
    evaluation_split: str | None = "development"
    internal_validation_fraction: float = 0.0
    max_length: int = 512
    stride: int = 64
    train_batch_size: int = 8
    evaluation_batch_size: int = 16
    epochs: float = 3.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    preprocessing_workers: int = 1
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    use_cpu: bool = False
    resume_from_checkpoint: Path | None = None
    overwrite_output: bool = False
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.revision.strip():
            raise ValueError("revision must pin a model commit or immutable release")
        if not self.train_split.strip():
            raise ValueError("train_split must be non-empty")
        if self.evaluation_split is not None and not self.evaluation_split.strip():
            raise ValueError("evaluation_split must be non-empty when provided")
        if self.evaluation_split == self.train_split:
            raise ValueError("evaluation_split must differ from train_split")
        if self.internal_validation_fraction < 0.0 or self.internal_validation_fraction >= 1.0:
            raise ValueError("internal_validation_fraction must be in [0, 1)")
        if self.internal_validation_fraction and self.evaluation_split is not None:
            raise ValueError(
                "internal_validation_fraction and evaluation_split are mutually exclusive"
            )
        if self.max_length < 8:
            raise ValueError("max_length must be at least 8")
        if self.stride < 0 or self.stride >= self.max_length - 2:
            raise ValueError("stride must fit inside max_length")
        if self.train_batch_size < 1 or self.evaluation_batch_size < 1:
            raise ValueError("batch sizes must be at least 1")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.preprocessing_workers < 1:
            raise ValueError("preprocessing_workers must be at least 1")
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot both be enabled")

    def to_dict(self) -> dict[str, Any]:
        """Return all behavior-affecting values for the experiment manifest."""

        return {
            "dataset_path": str(self.dataset_path),
            "dataset_manifest_path": str(self.dataset_manifest_path),
            "output_dir": str(self.output_dir),
            "model_id": self.model_id,
            "revision": self.revision,
            "train_split": self.train_split,
            "evaluation_split": self.evaluation_split,
            "internal_validation_fraction": self.internal_validation_fraction,
            "max_length": self.max_length,
            "stride": self.stride,
            "train_batch_size": self.train_batch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "preprocessing_workers": self.preprocessing_workers,
            "seed": self.seed,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "use_cpu": self.use_cpu,
            "resume_from_checkpoint": (
                None
                if self.resume_from_checkpoint is None
                else str(self.resume_from_checkpoint)
            ),
            "overwrite_output": self.overwrite_output,
            "cache_dir": None if self.cache_dir is None else str(self.cache_dir),
        }
