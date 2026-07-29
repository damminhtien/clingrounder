"""CLI commands for model-dataset validation and local model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, cast

from medical_kg_nlp.training import (
    CausalQLoRARunSpec,
    TokenClassifierRunSpec,
    TokenClassifierTrainingConfig,
    assert_local_gpu_runtime,
    build_dapt_corpus,
    finalize_causal_qlora_artifact,
    inspect_causal_qlora_inputs,
    inspect_local_runtime,
    inspect_token_classifier_training_inputs,
    load_causal_qlora_run_spec,
    load_dapt_corpus_build_spec,
    load_token_classifier_run_spec,
    train_causal_qlora,
    train_huggingface_token_classifier,
    verify_token_classifier_run_artifact,
    verify_saved_token_classifier,
)
from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.run_output import collect_git_metadata

__all__ = [
    "build_dapt_corpus_run",
    "finalize_causal_qlora_run",
    "inspect_causal_qlora_run",
    "inspect_token_classifier_run",
    "train_causal_qlora_run",
    "train_token_classifier",
    "train_token_classifier_run",
    "validate_token_dataset",
]


def build_dapt_corpus_run(args: argparse.Namespace) -> int:
    """Build physically separated, source-pinned DAPT corpus lanes."""

    manifest = build_dapt_corpus(load_dapt_corpus_build_spec(args.config))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def finalize_causal_qlora_run(args: argparse.Namespace) -> int:
    """Finalize a completed adapter created from a source archive."""

    spec = load_causal_qlora_run_spec(args.config)
    report = finalize_causal_qlora_artifact(
        spec.training.output_dir,
        model_id=spec.training.model_id,
        model_revision=spec.training.revision,
        parameter_count=spec.training.parameter_count,
        run_spec_path=spec.config_path,
        run_spec_sha256=sha256_file(spec.config_path),
        environment_lock_path=spec.environment_lock_path,
        environment_lock_sha256=spec.environment_lock_sha256,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def inspect_causal_qlora_run(args: argparse.Namespace) -> int:
    """Validate a pinned QLoRA stage without importing model frameworks."""

    spec = load_causal_qlora_run_spec(args.config)
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
                    "install_command": [
                        "uv",
                        "sync",
                        "--frozen",
                        "--extra",
                        "ml",
                    ],
                },
                "datasets": inspect_causal_qlora_inputs(spec.training),
                "model": {
                    "model_id": spec.training.model_id,
                    "revision": spec.training.revision,
                    "parameter_count": spec.training.parameter_count,
                    "source_url": spec.model_source_url,
                    "license": spec.model_license,
                },
                "runtime_requirements": spec.runtime.to_dict(),
                "local_runtime": inspect_local_runtime(spec.runtime),
                "trained_artifact": _inspect_causal_qlora_artifact(spec),
                "commands": {
                    "working_directory": "run_root",
                    "prefetch": list(spec.prefetch_command),
                    "train": [
                        "medical-kg",
                        "model",
                        "train-causal-qlora-run",
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


def train_causal_qlora_run(args: argparse.Namespace) -> int:
    """Execute one QLoRA stage after dataset, lockfile, and GPU validation."""

    spec = load_causal_qlora_run_spec(args.config)
    gpu_runtime = assert_local_gpu_runtime(spec.runtime)
    output_override = _run_root_path(spec, getattr(args, "output_dir", None))
    resume_checkpoint = _run_root_path(
        spec,
        getattr(args, "resume_from_checkpoint", None),
    )
    max_steps = getattr(args, "max_steps", None)
    manifest = dict(
        train_causal_qlora(
            spec.training,
            resume_from_checkpoint=resume_checkpoint,
            max_steps_override=max_steps,
            output_dir_override=output_override,
        )
    )
    output_dir = (
        spec.training.output_dir if output_override is None else output_override
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
    manifest["gpu_runtime"] = gpu_runtime
    manifest["purpose"] = "smoke" if max_steps is not None else "training"
    manifest["submission_eligible"] = max_steps is None
    write_json(output_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "trained",
                "run_id": spec.run_id,
                "manifest": spec.relative_path(output_dir / "run_manifest.json"),
                "metrics": manifest["metrics"],
                "gpu_runtime": gpu_runtime,
                "submission_eligible": manifest["submission_eligible"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_root_path(
    spec: CausalQLoRARunSpec,
    value: str | None,
) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (spec.run_root / path).resolve()
    spec.relative_path(resolved)
    return resolved


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
    return dict(verify_token_classifier_run_artifact(spec))


def _inspect_causal_qlora_artifact(
    spec: CausalQLoRARunSpec,
) -> dict[str, object]:
    """Report an adapter without importing PEFT or trusting an incomplete run."""

    manifest_path = spec.training.output_dir / "run_manifest.json"
    adapter_config = spec.training.output_dir / "final-adapter" / "adapter_config.json"
    if not manifest_path.is_file() or not adapter_config.is_file():
        return {
            "status": "absent",
            "manifest": spec.relative_path(manifest_path),
        }
    return {
        "status": "present",
        "manifest": spec.relative_path(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "adapter_config_sha256": sha256_file(adapter_config),
    }


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
