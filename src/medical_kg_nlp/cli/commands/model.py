"""CLI commands for model-dataset validation and local model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical_kg_nlp.training import (
    TokenClassifierTrainingConfig,
    inspect_token_classifier_training_inputs,
    train_huggingface_token_classifier,
)

__all__ = ["train_token_classifier", "validate_token_dataset"]


def validate_token_dataset(args: argparse.Namespace) -> int:
    """Validate dataset identity and split label compatibility without ML imports."""

    config = _config_from_args(args, output_dir=Path("."))
    summary, vocabulary = inspect_token_classifier_training_inputs(config)
    print(
        json.dumps(
            {
                "status": "valid",
                "dataset": summary.to_dict(),
                "label_vocabulary": list(vocabulary),
                "train_split": config.train_split,
                "evaluation_split": config.evaluation_split,
                "model": {
                    "model_id": config.model_id,
                    "revision": config.revision,
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def train_token_classifier(args: argparse.Namespace) -> int:
    """Train one locally cached, revision-pinned Hugging Face NER model."""

    config = _config_from_args(args, output_dir=Path(args.output_dir))
    manifest = train_huggingface_token_classifier(config)
    print(
        json.dumps(
            {
                "status": "trained",
                "manifest": str(config.output_dir / "run_manifest.json"),
                "model": manifest["model"],
                "metrics": manifest["metrics"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _config_from_args(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> TokenClassifierTrainingConfig:
    internal_validation_fraction = float(
        getattr(args, "internal_validation_fraction", 0.0)
    )
    evaluation_split = (
        None
        if args.no_evaluation or internal_validation_fraction
        else str(args.evaluation_split)
    )
    return TokenClassifierTrainingConfig(
        dataset_path=Path(args.dataset),
        dataset_manifest_path=Path(args.dataset_manifest),
        output_dir=output_dir,
        model_id=str(getattr(args, "model_id", "validation-only")),
        revision=str(getattr(args, "revision", "validation-only")),
        train_split=str(args.train_split),
        evaluation_split=evaluation_split,
        internal_validation_fraction=internal_validation_fraction,
        max_length=int(getattr(args, "max_length", 512)),
        stride=int(getattr(args, "stride", 64)),
        train_batch_size=int(getattr(args, "train_batch_size", 8)),
        evaluation_batch_size=int(getattr(args, "evaluation_batch_size", 16)),
        epochs=float(getattr(args, "epochs", 3.0)),
        learning_rate=float(getattr(args, "learning_rate", 2e-5)),
        weight_decay=float(getattr(args, "weight_decay", 0.01)),
        warmup_ratio=float(getattr(args, "warmup_ratio", 0.1)),
        gradient_accumulation_steps=int(
            getattr(args, "gradient_accumulation_steps", 1)
        ),
        preprocessing_workers=int(getattr(args, "preprocessing_workers", 1)),
        seed=int(getattr(args, "seed", 42)),
        fp16=bool(getattr(args, "fp16", False)),
        bf16=bool(getattr(args, "bf16", False)),
        use_cpu=bool(getattr(args, "cpu", False)),
        resume_from_checkpoint=(
            None
            if getattr(args, "resume_from_checkpoint", None) is None
            else Path(args.resume_from_checkpoint)
        ),
        overwrite_output=bool(getattr(args, "overwrite_output", False)),
        cache_dir=(
            None if getattr(args, "cache_dir", None) is None else Path(args.cache_dir)
        ),
    )
