"""Pinned, executable run specifications for Linux/CUDA token classifiers."""

from __future__ import annotations

import importlib
import importlib.util
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.adapters.huggingface.runtime import OptionalModelDependencyError
from medical_kg_nlp.training.config import TokenClassifierTrainingConfig
from medical_kg_nlp.utils.io import read_yaml

__all__ = [
    "GPURequirements",
    "TokenClassifierRunSpec",
    "assert_local_gpu_runtime",
    "inspect_local_runtime",
    "load_token_classifier_run_spec",
]

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class GPURequirements:
    """Minimum runtime contract checked before a model job starts."""

    operating_system: str = "linux"
    accelerator: str = "cuda"
    minimum_devices: int = 1
    minimum_vram_gib: float = 16.0
    minimum_compute_capability: tuple[int, int] = (8, 0)
    precision: str = "bf16"

    def __post_init__(self) -> None:
        if self.operating_system != "linux" or self.accelerator != "cuda":
            raise ValueError("Token-classifier GPU runs currently require Linux/CUDA")
        if self.minimum_devices < 1 or self.minimum_vram_gib <= 0:
            raise ValueError("GPU device and VRAM requirements must be positive")
        if self.minimum_compute_capability < (1, 0):
            raise ValueError("CUDA compute capability is invalid")
        if self.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("GPU precision must be bf16, fp16, or fp32")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operating_system": self.operating_system,
            "accelerator": self.accelerator,
            "minimum_devices": self.minimum_devices,
            "minimum_vram_gib": self.minimum_vram_gib,
            "minimum_compute_capability": list(self.minimum_compute_capability),
            "precision": self.precision,
        }


@dataclass(frozen=True)
class TokenClassifierRunSpec:
    """One immutable dataset/checkpoint/training/runtime experiment identity."""

    schema_version: str
    run_id: str
    training: TokenClassifierTrainingConfig
    runtime: GPURequirements
    model_source_url: str
    model_license: str

    def __post_init__(self) -> None:
        if self.schema_version != "token-classifier-run.v1":
            raise ValueError("Unsupported token-classifier run schema")
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        # MODEL: a full commit SHA prevents a mutable branch from changing the checkpoint.
        if _COMMIT_SHA.fullmatch(self.training.revision) is None:
            raise ValueError("Model revision must be a full 40-character commit SHA")
        expected_bf16 = self.runtime.precision == "bf16"
        expected_fp16 = self.runtime.precision == "fp16"
        if self.training.bf16 != expected_bf16 or self.training.fp16 != expected_fp16:
            raise ValueError("Training precision does not match the GPU runtime contract")
        if self.training.use_cpu:
            raise ValueError("A Linux/GPU run spec cannot enable use_cpu")

    @property
    def prefetch_command(self) -> tuple[str, ...]:
        """Return an explicit networked prefetch command for the pinned checkpoint."""

        return (
            "hf",
            "download",
            self.training.model_id,
            "--revision",
            self.training.revision,
        )

    @property
    def train_command(self) -> tuple[str, ...]:
        return (
            "medical-kg",
            "model",
            "train-token-classifier-run",
            "--config",
            f"configs/models/{self.run_id}.yaml",
        )


def load_token_classifier_run_spec(path: str | Path) -> TokenClassifierRunSpec:
    """Parse one strict YAML run spec without importing an ML framework."""

    raw = read_yaml(path)
    dataset = _mapping(raw, "dataset")
    model = _mapping(raw, "model")
    training = _mapping(raw, "training")
    runtime = _mapping(raw, "runtime")
    capability = runtime.get("minimum_compute_capability", [8, 0])
    if not isinstance(capability, list) or len(capability) != 2:
        raise ValueError("minimum_compute_capability must be [major, minor]")
    precision = str(runtime.get("precision", "bf16"))
    config = TokenClassifierTrainingConfig(
        dataset_path=Path(_required_string(dataset, "path")),
        dataset_manifest_path=Path(_required_string(dataset, "manifest")),
        output_dir=Path(_required_string(training, "output_dir")),
        model_id=_required_string(model, "model_id"),
        revision=_required_string(model, "revision"),
        train_split=str(dataset.get("train_split", "train")),
        evaluation_split=_optional_string(dataset.get("evaluation_split")),
        internal_validation_fraction=float(
            dataset.get("internal_validation_fraction", 0.0)
        ),
        max_length=int(training.get("max_length", 512)),
        stride=int(training.get("stride", 64)),
        train_batch_size=int(training.get("train_batch_size", 8)),
        evaluation_batch_size=int(training.get("evaluation_batch_size", 16)),
        epochs=float(training.get("epochs", 3.0)),
        learning_rate=float(training.get("learning_rate", 2e-5)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        warmup_ratio=float(training.get("warmup_ratio", 0.1)),
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 1)
        ),
        preprocessing_workers=int(training.get("preprocessing_workers", 1)),
        seed=int(training.get("seed", 42)),
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        use_cpu=False,
        overwrite_output=bool(training.get("overwrite_output", False)),
        cache_dir=(
            None
            if training.get("cache_dir") is None
            else Path(str(training["cache_dir"]))
        ),
    )
    return TokenClassifierRunSpec(
        schema_version=_required_string(raw, "schema_version"),
        run_id=_required_string(raw, "run_id"),
        training=config,
        runtime=GPURequirements(
            operating_system=str(runtime.get("operating_system", "linux")),
            accelerator=str(runtime.get("accelerator", "cuda")),
            minimum_devices=int(runtime.get("minimum_devices", 1)),
            minimum_vram_gib=float(runtime.get("minimum_vram_gib", 16.0)),
            minimum_compute_capability=(int(capability[0]), int(capability[1])),
            precision=precision,
        ),
        model_source_url=_required_string(model, "source_url"),
        model_license=_required_string(model, "license"),
    )


def inspect_local_runtime(requirements: GPURequirements) -> dict[str, Any]:
    """Describe the host without importing Torch or allocating a CUDA context."""

    system = platform.system().lower()
    return {
        "operating_system": system,
        "machine": platform.machine(),
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "static_os_compatible": system == requirements.operating_system,
        "gpu_check": "deferred_until_train",
    }


def assert_local_gpu_runtime(requirements: GPURequirements) -> dict[str, Any]:
    """Fail before model loading when Linux/CUDA requirements are not satisfied."""

    system = platform.system().lower()
    if system != requirements.operating_system:
        raise RuntimeError(
            f"Run requires {requirements.operating_system}, current host is {system}"
        )
    try:
        torch = importlib.import_module("torch")
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalModelDependencyError(
            "GPU training requires the local 'ml' extra: uv sync --extra ml"
        ) from error
    cuda = torch.cuda
    if not bool(cuda.is_available()):
        raise RuntimeError("CUDA is not available on this host")
    device_count = int(cuda.device_count())
    if device_count < requirements.minimum_devices:
        raise RuntimeError(
            f"Run requires {requirements.minimum_devices} CUDA device(s), found {device_count}"
        )
    properties = cuda.get_device_properties(0)
    capability = tuple(int(value) for value in cuda.get_device_capability(0))
    vram_gib = float(properties.total_memory) / (1024**3)
    if capability < requirements.minimum_compute_capability:
        raise RuntimeError(
            f"GPU compute capability {capability} is below "
            f"{requirements.minimum_compute_capability}"
        )
    if vram_gib < requirements.minimum_vram_gib:
        raise RuntimeError(
            f"GPU VRAM {vram_gib:.2f} GiB is below {requirements.minimum_vram_gib:.2f} GiB"
        )
    if requirements.precision == "bf16" and not bool(cuda.is_bf16_supported()):
        raise RuntimeError("Configured GPU does not support BF16")
    return {
        "operating_system": system,
        "torch": str(getattr(torch, "__version__", "unknown")),
        "cuda": str(getattr(torch.version, "cuda", "unknown")),
        "device_count": device_count,
        "device_name": str(properties.name),
        "compute_capability": list(capability),
        "vram_gib": round(vram_gib, 3),
        "precision": requirements.precision,
    }


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Run spec {key} must be a mapping")
    return value


def _required_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Run spec {key} must be a non-empty string")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Optional run-spec string must be non-empty")
    return value
