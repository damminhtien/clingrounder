"""Pinned configuration for reproducible local token-classifier training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = ["TokenClassifierTrainingConfig"]


@dataclass(frozen=True)
class TokenClassifierTrainingConfig:
    """Dataset, model identity, and bounded training hyperparameters."""

    dataset_path: Path
    dataset_manifest_path: Path
    output_dir: Path
    model_id: str
    revision: str
    initialization_model_path: Path | None = None
    initialization_model_fingerprint: str | None = None
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
    full_determinism: bool = False
    resume_from_checkpoint: Path | None = None
    overwrite_output: bool = False
    cache_dir: Path | None = None
    unaligned_span_policy: Literal["error", "mask"] = "error"

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.revision.strip():
            raise ValueError("revision must pin a model commit or immutable release")
        if (self.initialization_model_path is None) != (
            self.initialization_model_fingerprint is None
        ):
            raise ValueError(
                "initialization_model_path and initialization_model_fingerprint "
                "must be provided together"
            )
        if (
            self.initialization_model_fingerprint is not None
            and (
                len(self.initialization_model_fingerprint) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.initialization_model_fingerprint
                )
            )
        ):
            raise ValueError(
                "initialization_model_fingerprint must be a lowercase SHA-256 value"
            )
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
        if self.unaligned_span_policy not in {"error", "mask"}:
            raise ValueError("unaligned_span_policy must be 'error' or 'mask'")

    def to_dict(self, *, path_root: Path | None = None) -> dict[str, Any]:
        """Return behavior-affecting values with optionally portable paths.

        ``path_root`` is used by checked-in run specifications. It prevents an
        otherwise reproducible manifest from embedding the workstation checkout
        path while preserving resolved absolute paths for runtime IO.
        """

        return {
            "dataset_path": _manifest_path(self.dataset_path, root=path_root),
            "dataset_manifest_path": _manifest_path(
                self.dataset_manifest_path,
                root=path_root,
            ),
            "output_dir": _manifest_path(self.output_dir, root=path_root),
            "model_id": self.model_id,
            "revision": self.revision,
            "initialization_model_path": (
                None
                if self.initialization_model_path is None
                else _manifest_path(self.initialization_model_path, root=path_root)
            ),
            "initialization_model_fingerprint": self.initialization_model_fingerprint,
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
            "full_determinism": self.full_determinism,
            "resume_from_checkpoint": (
                None
                if self.resume_from_checkpoint is None
                else _manifest_path(self.resume_from_checkpoint, root=path_root)
            ),
            "overwrite_output": self.overwrite_output,
            "cache_dir": (
                None if self.cache_dir is None else _manifest_path(self.cache_dir, root=path_root)
            ),
            "unaligned_span_policy": self.unaligned_span_policy,
        }


def _manifest_path(path: Path, *, root: Path | None) -> str:
    if root is None:
        return str(path)
    resolved_root = root.resolve()
    try:
        # INVARIANT: portable manifests cannot reference data outside the run root.
        return path.resolve().relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError(f"Training path escapes manifest root: {path}") from error
