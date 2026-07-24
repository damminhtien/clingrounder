"""Pinned, executable run specifications for Linux/CUDA token classifiers."""

from __future__ import annotations

import importlib
import importlib.util
import json
import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from medical_kg_nlp.adapters.huggingface.runtime import OptionalModelDependencyError
from medical_kg_nlp.training.config import TokenClassifierTrainingConfig
from medical_kg_nlp.training.huggingface_token_classifier import (
    verify_token_classifier_artifact,
)
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.io import read_yaml

__all__ = [
    "GPURequirements",
    "TokenClassifierRunSpec",
    "assert_local_gpu_runtime",
    "inspect_local_runtime",
    "load_token_classifier_run_spec",
    "verify_token_classifier_run_artifact",
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
    config_path: Path
    run_root: Path
    training: TokenClassifierTrainingConfig
    runtime: GPURequirements
    environment_lock_path: Path
    environment_lock_sha256: str
    model_source_url: str
    model_license: str

    def __post_init__(self) -> None:
        if self.schema_version != "token-classifier-run.v2":
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
        if not self.training.full_determinism:
            raise ValueError("A reproducible Linux/GPU run must enable full_determinism")
        if not self.environment_lock_path.is_file():
            raise ValueError(
                f"Environment lock file does not exist: {self.environment_lock_path}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.environment_lock_sha256):
            raise ValueError("Environment lock SHA-256 must be 64 lowercase hexadecimal characters")
        actual_lock_sha256 = sha256_file(self.environment_lock_path)
        if actual_lock_sha256 != self.environment_lock_sha256:
            raise ValueError(
                "Environment lock SHA-256 mismatch: "
                f"expected {self.environment_lock_sha256}, got {actual_lock_sha256}"
            )

    @property
    def config_relative_path(self) -> str:
        """Return the run-spec path without leaking the checkout location."""

        return self.relative_path(self.config_path)

    def relative_path(self, path: str | Path) -> str:
        """Project one resolved runtime path into the portable run-root namespace."""

        try:
            # INVARIANT: all scientific inputs and outputs belong to one movable tree.
            return Path(path).resolve().relative_to(self.run_root).as_posix()
        except ValueError as error:
            raise ValueError(f"Run path escapes declared run_root: {path}") from error

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
            self.config_relative_path,
        )


def load_token_classifier_run_spec(path: str | Path) -> TokenClassifierRunSpec:
    """Parse one strict YAML run spec without importing an ML framework."""

    config_path = Path(path).resolve()
    raw = read_yaml(config_path)
    schema_version = _required_string(raw, "schema_version")
    if schema_version != "token-classifier-run.v2":
        raise ValueError("Unsupported token-classifier run schema")
    run_root = _resolve_run_root(
        config_path,
        _required_string(raw, "run_root"),
    )
    _relative_to_run_root(config_path, run_root)
    dataset = _mapping(raw, "dataset")
    environment = _mapping(raw, "environment")
    model = _mapping(raw, "model")
    training = _mapping(raw, "training")
    runtime = _mapping(raw, "runtime")
    capability = runtime.get("minimum_compute_capability", [8, 0])
    if not isinstance(capability, list) or len(capability) != 2:
        raise ValueError("minimum_compute_capability must be [major, minor]")
    precision = str(runtime.get("precision", "bf16"))
    config = TokenClassifierTrainingConfig(
        dataset_path=_resolve_run_path(
            run_root,
            _required_string(dataset, "path"),
            field="dataset.path",
        ),
        dataset_manifest_path=_resolve_run_path(
            run_root,
            _required_string(dataset, "manifest"),
            field="dataset.manifest",
        ),
        output_dir=_resolve_run_path(
            run_root,
            _required_string(training, "output_dir"),
            field="training.output_dir",
        ),
        model_id=_required_string(model, "model_id"),
        revision=_required_string(model, "revision"),
        train_split=str(dataset.get("train_split", "train")),
        evaluation_split=_optional_string(dataset.get("evaluation_split")),
        internal_validation_fraction=float(dataset.get("internal_validation_fraction", 0.0)),
        max_length=int(training.get("max_length", 512)),
        stride=int(training.get("stride", 64)),
        train_batch_size=int(training.get("train_batch_size", 8)),
        evaluation_batch_size=int(training.get("evaluation_batch_size", 16)),
        epochs=float(training.get("epochs", 3.0)),
        learning_rate=float(training.get("learning_rate", 2e-5)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        warmup_ratio=float(training.get("warmup_ratio", 0.1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        preprocessing_workers=int(training.get("preprocessing_workers", 1)),
        seed=int(training.get("seed", 42)),
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        use_cpu=False,
        full_determinism=_boolean(training, "full_determinism", default=False),
        overwrite_output=_boolean(training, "overwrite_output", default=False),
        cache_dir=(
            None
            if training.get("cache_dir") is None
            else _resolve_run_path(
                run_root,
                str(training["cache_dir"]),
                field="training.cache_dir",
            )
        ),
        unaligned_span_policy=_alignment_policy(training.get("unaligned_span_policy", "error")),
    )
    return TokenClassifierRunSpec(
        schema_version=schema_version,
        run_id=_required_string(raw, "run_id"),
        config_path=config_path,
        run_root=run_root,
        training=config,
        runtime=GPURequirements(
            operating_system=str(runtime.get("operating_system", "linux")),
            accelerator=str(runtime.get("accelerator", "cuda")),
            minimum_devices=int(runtime.get("minimum_devices", 1)),
            minimum_vram_gib=float(runtime.get("minimum_vram_gib", 16.0)),
            minimum_compute_capability=(int(capability[0]), int(capability[1])),
            precision=precision,
        ),
        environment_lock_path=_resolve_run_path(
            run_root,
            _required_string(environment, "lock_path"),
            field="environment.lock_path",
        ),
        environment_lock_sha256=_required_string(environment, "lock_sha256"),
        model_source_url=_required_string(model, "source_url"),
        model_license=_required_string(model, "license"),
    )


def verify_token_classifier_run_artifact(
    spec: TokenClassifierRunSpec,
) -> dict[str, Any]:
    """Verify a trained GPU artifact against every immutable run input.

    This check deliberately avoids importing Torch or Transformers. A transferred checkpoint must
    pass it before development inference so a stale model directory cannot be calibrated against a
    newer dataset, run spec, or dependency lock.
    """

    manifest_path = spec.training.output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Token-classifier run manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _json_mapping(payload, "token-classifier run manifest")
    if manifest.get("submission_eligible") is False or manifest.get("purpose") == "cpu_smoke":
        raise ValueError("CPU-smoke token classifier cannot enter model selection")

    model = _json_mapping(manifest.get("model"), "token-classifier model metadata")
    expected_model_dir = spec.training.output_dir / "final-model"
    if model.get("output") != spec.relative_path(expected_model_dir):
        raise ValueError("Training manifest model output does not match the run specification")
    if model.get("model_id") != spec.training.model_id:
        raise ValueError("Training manifest model ID does not match the run specification")
    if model.get("revision") != spec.training.revision:
        raise ValueError("Training manifest model revision does not match the run specification")

    run_spec = _json_mapping(
        manifest.get("run_spec"),
        "token-classifier run-spec provenance",
    )
    if run_spec.get("sha256") != sha256_file(spec.config_path):
        raise ValueError("Training manifest run-spec SHA-256 does not match")
    if run_spec.get("run_id") != spec.run_id:
        raise ValueError("Training manifest run ID does not match")

    environment = _json_mapping(
        manifest.get("environment"),
        "token-classifier environment provenance",
    )
    if environment.get("lock_sha256") != spec.environment_lock_sha256:
        raise ValueError("Training manifest environment lock SHA-256 does not match")
    if manifest.get("dataset_manifest_sha256") != sha256_file(
        spec.training.dataset_manifest_path
    ):
        raise ValueError("Training manifest dataset-manifest SHA-256 does not match")

    gpu_runtime = _json_mapping(
        manifest.get("gpu_runtime"),
        "token-classifier GPU provenance",
    )
    if gpu_runtime.get("precision") != spec.runtime.precision:
        raise ValueError("Training manifest GPU precision does not match")

    artifact = verify_token_classifier_artifact(expected_model_dir, manifest_path)
    return {
        "status": "verified",
        "manifest": spec.relative_path(manifest_path),
        "manifest_sha256": artifact["manifest_sha256"],
        "model": spec.relative_path(expected_model_dir),
        "model_id": spec.training.model_id,
        "revision": spec.training.revision,
        "fingerprint": artifact["fingerprint"],
        "gpu_runtime": dict(gpu_runtime),
    }


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


def _json_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
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


def _boolean(raw: dict[str, Any], key: str, *, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Run spec {key} must be a boolean")
    return value


def _alignment_policy(value: object) -> Literal["error", "mask"]:
    if value == "error":
        return "error"
    if value == "mask":
        return "mask"
    raise ValueError("training.unaligned_span_policy must be 'error' or 'mask'")


def _resolve_run_root(config_path: Path, raw_root: str) -> Path:
    if Path(raw_root).is_absolute() or PureWindowsPath(raw_root).is_absolute():
        raise ValueError("run_root must be relative to the run specification")
    root = (config_path.parent / raw_root).resolve()
    if not root.is_dir():
        raise ValueError(f"run_root does not exist: {raw_root}")
    return root


def _resolve_run_path(root: Path, raw_path: str, *, field: str) -> Path:
    if Path(raw_path).is_absolute() or PureWindowsPath(raw_path).is_absolute():
        raise ValueError(f"{field} must be relative to run_root")
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes declared run_root") from error
    return path


def _relative_to_run_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Run specification must be inside run_root") from error
