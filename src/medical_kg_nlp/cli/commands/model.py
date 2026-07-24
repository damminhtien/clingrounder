"""CLI commands for model-dataset validation and local model training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from medical_kg_nlp.training import (
    TokenClassifierRunSpec,
    TokenClassifierTrainingConfig,
    assert_local_gpu_runtime,
    inspect_local_runtime,
    inspect_token_classifier_training_inputs,
    load_token_classifier_run_spec,
    train_huggingface_token_classifier,
    verify_token_classifier_artifact,
    verify_saved_token_classifier,
)
from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.run_output import collect_git_metadata

__all__ = [
    "inspect_token_classifier_run",
    "train_token_classifier",
    "train_token_classifier_run",
    "validate_token_dataset",
]


def inspect_token_classifier_run(args: argparse.Namespace) -> int:
    """Validate a run spec and render exact prefetch/train commands."""

    spec = load_token_classifier_run_spec(args.config)
    summary, vocabulary = inspect_token_classifier_training_inputs(spec.training)
    print(
        json.dumps(
            {
                "status": "validated_not_executed",
                "run_id": spec.run_id,
                "run_spec": {
                    "path": spec.config_relative_path,
                    "path_base": "run_root",
                    "sha256": sha256_file(spec.config_path),
                },
                "environment": {
                    "lock_path": spec.relative_path(spec.environment_lock_path),
                    "lock_sha256": spec.environment_lock_sha256,
                    "install_command": ["uv", "sync", "--frozen", "--extra", "ml"],
                },
                "dataset": summary.to_dict(),
                "label_vocabulary": list(vocabulary),
                "model": {
                    "model_id": spec.training.model_id,
                    "revision": spec.training.revision,
                    "source_url": spec.model_source_url,
                    "license": spec.model_license,
                },
                "runtime_requirements": spec.runtime.to_dict(),
                "local_runtime": inspect_local_runtime(spec.runtime),
                "trained_artifact": _inspect_trained_artifact(spec),
                "commands": {
                    "working_directory": "run_root",
                    "prefetch": list(spec.prefetch_command),
                    "train": [
                        "medical-kg",
                        "model",
                        "train-token-classifier-run",
                        "--config",
                        spec.config_relative_path,
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def train_token_classifier_run(args: argparse.Namespace) -> int:
    """Execute a run spec only after its Linux/CUDA gate passes."""

    spec = load_token_classifier_run_spec(args.config)
    gpu_runtime = assert_local_gpu_runtime(spec.runtime)
    manifest = dict(
        train_huggingface_token_classifier(
            spec.training,
            manifest_root=spec.run_root,
        )
    )
    manifest["run_spec"] = {
        "path": spec.config_relative_path,
        "path_base": "run_root",
        "sha256": sha256_file(spec.config_path),
        "run_id": spec.run_id,
    }
    manifest["environment"] = {
        "lock_path": spec.relative_path(spec.environment_lock_path),
        "lock_sha256": spec.environment_lock_sha256,
        "install_command": ["uv", "sync", "--frozen", "--extra", "ml"],
    }
    manifest["source_control"] = collect_git_metadata()
    manifest["gpu_runtime"] = gpu_runtime
    write_json(spec.training.output_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "trained",
                "run_id": spec.run_id,
                "manifest": spec.relative_path(spec.training.output_dir / "run_manifest.json"),
                "model": manifest["model"],
                "metrics": manifest["metrics"],
                "gpu_runtime": gpu_runtime,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _inspect_trained_artifact(spec: TokenClassifierRunSpec) -> dict[str, object]:
    """Verify a returned model without importing Torch or Transformers."""

    manifest_path = spec.training.output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return {
            "status": "absent",
            "manifest": spec.relative_path(manifest_path),
        }
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _json_mapping(payload, "training manifest")
    model = _json_mapping(manifest.get("model"), "training manifest model")
    run_spec = _json_mapping(manifest.get("run_spec"), "training manifest run_spec")
    environment = _json_mapping(
        manifest.get("environment"),
        "training manifest environment",
    )

    expected_model_path = spec.relative_path(spec.training.output_dir / "final-model")
    if model.get("output") != expected_model_path:
        raise ValueError("Training manifest model output does not match the run specification")
    artifact = verify_token_classifier_artifact(
        spec.training.output_dir / "final-model",
        manifest_path,
    )
    if run_spec.get("sha256") != sha256_file(spec.config_path):
        raise ValueError("Training manifest run-spec SHA-256 does not match")
    if environment.get("lock_sha256") != spec.environment_lock_sha256:
        raise ValueError("Training manifest environment lock SHA-256 does not match")
    if manifest.get("dataset_manifest_sha256") != sha256_file(
        spec.training.dataset_manifest_path
    ):
        raise ValueError("Training manifest dataset-manifest SHA-256 does not match")
    return {
        "status": "verified",
        "manifest": spec.relative_path(manifest_path),
        "model": expected_model_path,
        "fingerprint": artifact["fingerprint"],
    }


def _json_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


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
    smoke_text_path = getattr(args, "cpu_smoke_text", None)
    if smoke_text_path and not config.use_cpu:
        raise ValueError("--cpu-smoke-text requires --cpu")
    manifest = dict(train_huggingface_token_classifier(config))
    if smoke_text_path:
        text_path = Path(smoke_text_path)
        # Keep newline semantics identical to pipeline input; Path.read_text does not normalize
        # newline bytes on this explicit UTF-8 path.
        source_text = text_path.read_bytes().decode("utf-8")
        verification = dict(
            verify_saved_token_classifier(
                config.output_dir / "final-model",
                source_text,
                max_length=config.max_length,
                stride=config.stride,
                batch_size=config.evaluation_batch_size,
            )
        )
        verification["source_path"] = str(text_path)
        manifest["cpu_smoke"] = verification
        manifest["purpose"] = "cpu_smoke"
        manifest["submission_eligible"] = False
        write_json(config.output_dir / "run_manifest.json", manifest)
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
    internal_validation_fraction = float(getattr(args, "internal_validation_fraction", 0.0))
    evaluation_split = (
        None if args.no_evaluation or internal_validation_fraction else str(args.evaluation_split)
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
        gradient_accumulation_steps=int(getattr(args, "gradient_accumulation_steps", 1)),
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
        cache_dir=(None if getattr(args, "cache_dir", None) is None else Path(args.cache_dir)),
        unaligned_span_policy=cast(
            Literal["error", "mask"],
            str(getattr(args, "unaligned_span_policy", "error")),
        ),
    )
