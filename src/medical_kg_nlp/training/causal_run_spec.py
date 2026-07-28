"""Pinned two-stage causal QLoRA run specifications."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.training.causal_instruction import CausalInstructionSource
from medical_kg_nlp.training.run_spec import GPURequirements
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.io import read_yaml

__all__ = [
    "CausalQLoRAConfig",
    "CausalQLoRARunSpec",
    "load_causal_qlora_run_spec",
]

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class CausalQLoRAConfig:
    """Framework-neutral controls for one resumable adapter-training stage."""

    train_sources: tuple[CausalInstructionSource, ...]
    evaluation_sources: tuple[CausalInstructionSource, ...]
    output_dir: Path
    model_id: str
    revision: str
    parameter_count: int
    initial_adapter_path: Path | None
    sample_seed: str
    max_length: int
    train_batch_size: int
    evaluation_batch_size: int
    gradient_accumulation_steps: int
    epochs: float
    max_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    logging_steps: int
    evaluation_steps: int
    save_steps: int
    save_total_limit: int
    seed: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    gradient_checkpointing: bool
    local_files_only: bool

    def __post_init__(self) -> None:
        if not self.train_sources:
            raise ValueError("QLoRA requires at least one training source")
        if _COMMIT_SHA.fullmatch(self.revision) is None:
            raise ValueError("QLoRA model revision must be a full commit SHA")
        if not 0 < self.parameter_count <= 9_000_000_000:
            raise ValueError("QLoRA model must respect the 9B parameter limit")
        if self.max_length < 256 or self.train_batch_size < 1:
            raise ValueError("QLoRA max_length and batch sizes must be positive")
        if self.evaluation_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("QLoRA evaluation and accumulation sizes must be positive")
        if self.epochs <= 0 or self.max_steps < -1 or self.max_steps == 0:
            raise ValueError("QLoRA epochs and max_steps are invalid")
        if self.learning_rate <= 0:
            raise ValueError("QLoRA epochs, max_steps, and learning_rate are invalid")
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("QLoRA warmup/dropout must be in [0, 1)")
        if self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("QLoRA rank and alpha must be positive")


@dataclass(frozen=True, slots=True)
class CausalQLoRARunSpec:
    """Portable scientific identity for one QLoRA stage."""

    schema_version: str
    run_id: str
    config_path: Path
    run_root: Path
    training: CausalQLoRAConfig
    runtime: GPURequirements
    environment_lock_path: Path
    environment_lock_sha256: str
    model_source_url: str
    model_license: str
    maximum_vast_cost_usd: float

    def __post_init__(self) -> None:
        if self.schema_version != "causal-qlora-run.v1":
            raise ValueError("Unsupported causal QLoRA run schema")
        if not self.run_id.strip():
            raise ValueError("QLoRA run_id must be non-empty")
        if not 0 < self.maximum_vast_cost_usd <= 6.0:
            raise ValueError("QLoRA Vast budget must be in (0, 6]")
        if not self.environment_lock_path.is_file():
            raise ValueError("QLoRA environment lock file is missing")
        if sha256_file(self.environment_lock_path) != self.environment_lock_sha256:
            raise ValueError("QLoRA environment lock SHA-256 mismatch")
        self.relative_path(self.config_path)
        self.relative_path(self.training.output_dir)
        if self.training.initial_adapter_path is not None:
            self.relative_path(self.training.initial_adapter_path)

    @property
    def config_relative_path(self) -> str:
        return self.relative_path(self.config_path)

    def relative_path(self, path: str | Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.run_root).as_posix()
        except ValueError as error:
            raise ValueError(f"QLoRA path escapes run_root: {path}") from error

    @property
    def prefetch_command(self) -> tuple[str, ...]:
        return (
            "hf",
            "download",
            self.training.model_id,
            "--revision",
            self.training.revision,
        )


def load_causal_qlora_run_spec(path: str | Path) -> CausalQLoRARunSpec:
    """Load and validate a QLoRA YAML without importing ML dependencies."""

    config_path = Path(path).resolve()
    raw = read_yaml(config_path)
    if raw.get("schema_version") != "causal-qlora-run.v1":
        raise ValueError("Unsupported causal QLoRA run schema")
    run_root = _resolve(config_path.parent, _required(raw, "run_root"))
    model = _mapping(raw, "model")
    data = _mapping(raw, "datasets")
    training = _mapping(raw, "training")
    runtime = _mapping(raw, "runtime")
    environment = _mapping(raw, "environment")
    remote = _mapping(raw, "remote")
    precision = str(runtime.get("precision", "bf16"))
    capability = runtime.get("minimum_compute_capability", [8, 0])
    if not isinstance(capability, list) or len(capability) != 2:
        raise ValueError("minimum_compute_capability must be [major, minor]")
    initial_adapter = training.get("initial_adapter_path")
    config = CausalQLoRAConfig(
        train_sources=_sources(run_root, data.get("train"), "datasets.train"),
        evaluation_sources=_sources(
            run_root,
            data.get("evaluation", []),
            "datasets.evaluation",
        ),
        output_dir=_resolve(run_root, _required(training, "output_dir")),
        model_id=_required(model, "model_id"),
        revision=_required(model, "revision"),
        parameter_count=int(model["parameter_count"]),
        initial_adapter_path=(
            None
            if initial_adapter is None
            else _resolve(run_root, str(initial_adapter))
        ),
        sample_seed=str(data.get("sample_seed", "causal-qlora-v1")),
        max_length=int(training.get("max_length", 4096)),
        train_batch_size=int(training.get("train_batch_size", 1)),
        evaluation_batch_size=int(training.get("evaluation_batch_size", 1)),
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 8)
        ),
        epochs=float(training.get("epochs", 1.0)),
        max_steps=int(training.get("max_steps", -1)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        warmup_ratio=float(training.get("warmup_ratio", 0.05)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        logging_steps=int(training.get("logging_steps", 1)),
        evaluation_steps=int(training.get("evaluation_steps", 20)),
        save_steps=int(training.get("save_steps", 20)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        seed=int(training.get("seed", 42)),
        lora_rank=int(training.get("lora_rank", 16)),
        lora_alpha=int(training.get("lora_alpha", 32)),
        lora_dropout=float(training.get("lora_dropout", 0.05)),
        gradient_checkpointing=bool(
            training.get("gradient_checkpointing", True)
        ),
        local_files_only=bool(runtime.get("local_files_only", True)),
    )
    return CausalQLoRARunSpec(
        schema_version=str(raw["schema_version"]),
        run_id=_required(raw, "run_id"),
        config_path=config_path,
        run_root=run_root,
        training=config,
        runtime=GPURequirements(
            operating_system=str(runtime.get("operating_system", "linux")),
            accelerator=str(runtime.get("accelerator", "cuda")),
            minimum_devices=int(runtime.get("minimum_devices", 1)),
            minimum_vram_gib=float(runtime.get("minimum_vram_gib", 20.0)),
            minimum_compute_capability=(int(capability[0]), int(capability[1])),
            precision=precision,
        ),
        environment_lock_path=_resolve(
            run_root,
            _required(environment, "lock_path"),
        ),
        environment_lock_sha256=_required(environment, "lock_sha256"),
        model_source_url=_required(model, "source_url"),
        model_license=_required(model, "license"),
        maximum_vast_cost_usd=float(
            remote.get("maximum_vast_cost_usd", 6.0)
        ),
    )


def _sources(
    run_root: Path,
    value: object,
    field: str,
) -> tuple[CausalInstructionSource, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    output: list[CausalInstructionSource] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be a mapping")
        output.append(
            CausalInstructionSource(
                path=_resolve(run_root, _required(item, "path")),
                sha256=_required(item, "sha256"),
                split=str(item.get("split", "train")),
                maximum_records=(
                    None
                    if item.get("maximum_records") is None
                    else int(item["maximum_records"])
                ),
                repeat=int(item.get("repeat", 1)),
                document_id_prefix=(
                    None
                    if item.get("document_id_prefix") is None
                    else str(item["document_id_prefix"])
                ),
            )
        )
    return tuple(output)


def _mapping(raw: dict[str, Any], field: str) -> dict[str, Any]:
    value = raw.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()
