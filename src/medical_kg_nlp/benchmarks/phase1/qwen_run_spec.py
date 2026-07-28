"""Pinned Qwen benchmark specifications with local model and Vast cost gates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.adapters.generative import (
    GenerationConfig,
    InferenceBudgetManifest,
    ModelBudgetEntry,
)
from medical_kg_nlp.utils.hashing import sha256_directory, sha256_file
from medical_kg_nlp.utils.io import read_yaml

__all__ = [
    "Phase1PeftAdapterSpec",
    "Phase1QwenRunSpec",
    "load_phase1_qwen_run_spec",
]

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class Phase1PeftAdapterSpec:
    """Pinned local PEFT adapter and the training evidence that produced it."""

    model: ModelBudgetEntry
    path: Path
    fingerprint: str
    provenance_manifest_path: Path
    provenance_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.model.kind != "adapter":
            raise ValueError("Qwen PEFT budget entry must use kind=adapter")
        if _SHA256.fullmatch(self.fingerprint) is None:
            raise ValueError("Qwen PEFT fingerprint must be a lowercase SHA-256")
        if _SHA256.fullmatch(self.provenance_manifest_sha256) is None:
            raise ValueError(
                "Qwen PEFT provenance manifest hash must be a lowercase SHA-256"
            )

    def to_dict(self, run_spec: Phase1QwenRunSpec) -> dict[str, Any]:
        """Serialize paths relative to the portable Qwen run root."""

        return {
            **self.model.to_dict(),
            "path": run_spec.relative_path(self.path),
            "fingerprint": self.fingerprint,
            "provenance": {
                "manifest": run_spec.relative_path(self.provenance_manifest_path),
                "manifest_sha256": self.provenance_manifest_sha256,
            },
        }

    def verify(self, base_model: ModelBudgetEntry) -> dict[str, Any]:
        """Verify adapter bytes, base compatibility, and training provenance."""

        if not self.path.is_dir():
            raise ValueError(f"Qwen PEFT adapter directory is absent: {self.path}")
        adapter_config_path = self.path / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise ValueError("Qwen PEFT adapter_config.json is absent")
        actual_fingerprint = sha256_directory(self.path)
        if actual_fingerprint != self.fingerprint:
            raise ValueError(
                "Qwen PEFT adapter fingerprint mismatch: "
                f"expected {self.fingerprint}, got {actual_fingerprint}"
            )
        if not self.provenance_manifest_path.is_file():
            raise ValueError("Qwen PEFT provenance manifest is absent")
        actual_manifest_sha256 = sha256_file(self.provenance_manifest_path)
        if actual_manifest_sha256 != self.provenance_manifest_sha256:
            raise ValueError(
                "Qwen PEFT provenance manifest SHA-256 mismatch: "
                f"expected {self.provenance_manifest_sha256}, "
                f"got {actual_manifest_sha256}"
            )

        adapter_config = _json_mapping(adapter_config_path, "PEFT adapter config")
        configured_base = adapter_config.get("base_model_name_or_path")
        if configured_base != base_model.model_id:
            raise ValueError(
                "Qwen PEFT adapter base model mismatch: "
                f"expected {base_model.model_id!r}, got {configured_base!r}"
            )
        provenance = _json_mapping(
            self.provenance_manifest_path,
            "PEFT provenance manifest",
        )
        if provenance.get("schema_version") != "causal-qlora-artifact.v1":
            raise ValueError("Unsupported Qwen PEFT provenance schema")
        trained_model = _as_mapping(provenance.get("model"), "trained model")
        trained_parameter_count = trained_model.get("parameter_count")
        if (
            trained_model.get("model_id") != base_model.model_id
            or trained_model.get("revision") != base_model.revision
            or not isinstance(trained_parameter_count, int)
            or trained_parameter_count != base_model.parameter_count
        ):
            raise ValueError(
                "Qwen PEFT provenance does not match the pinned base checkpoint"
            )
        source_control = _as_mapping(
            provenance.get("source_control"),
            "adapter source control",
        )
        if source_control.get("git_commit") != self.model.revision:
            raise ValueError(
                "Qwen PEFT budget revision does not match the training commit"
            )
        artifacts = _as_mapping(provenance.get("artifacts"), "adapter artifacts")
        expected_config_sha256 = artifacts.get("adapter_config_sha256")
        if expected_config_sha256 != sha256_file(adapter_config_path):
            raise ValueError(
                "Qwen PEFT adapter config does not match training provenance"
            )
        return {
            "status": "verified",
            "fingerprint": actual_fingerprint,
            "manifest_sha256": actual_manifest_sha256,
            "adapter_config_sha256": expected_config_sha256,
        }


@dataclass(frozen=True, slots=True)
class Phase1QwenRunSpec:
    """One portable extraction/adjudication experiment identity."""

    schema_version: str
    run_id: str
    config_path: Path
    run_root: Path
    model: ModelBudgetEntry
    model_source_url: str
    model_license: str
    adapter: Phase1PeftAdapterSpec | None
    budget: InferenceBudgetManifest
    dataset_path: Path
    dataset_manifest_path: Path
    device: str
    dtype: str
    local_files_only: bool
    max_window_characters: int
    window_overlap_characters: int
    structured_retries: int
    recall_generation: GenerationConfig
    targeted_generation: GenerationConfig
    adjudication_generation: GenerationConfig
    thresholds: dict[str, float]
    maximum_vast_cost_usd: float

    def __post_init__(self) -> None:
        if self.schema_version != "phase1-qwen-run.v1":
            raise ValueError("Unsupported Phase 1 Qwen run schema")
        if not self.run_id.strip():
            raise ValueError("Qwen run_id must be non-empty")
        if self.model not in self.budget.entries:
            raise ValueError("Primary Qwen model must be included in the inference budget")
        if self.adapter is not None and self.adapter.model not in self.budget.entries:
            raise ValueError("Qwen PEFT adapter must be included in the inference budget")
        if self.adapter is not None and not set(self.adapter.model.roles).issubset(
            self.model.roles
        ):
            raise ValueError("Qwen PEFT roles must be a subset of base-model roles")
        # INVARIANT: specs may be inspected before derived data is built, but paths cannot escape
        # the portable run root. Execution calls ``verify_dataset_inputs`` explicitly.
        self.relative_path(self.dataset_path)
        self.relative_path(self.dataset_manifest_path)
        if self.adapter is not None:
            self.relative_path(self.adapter.path)
            self.relative_path(self.adapter.provenance_manifest_path)
        if self.device not in {"cuda", "cpu", "mps"}:
            raise ValueError("Qwen device must be cuda, cpu, or mps")
        if self.dtype not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("Qwen dtype must be auto, bf16, fp16, or fp32")
        if self.maximum_vast_cost_usd <= 0 or self.maximum_vast_cost_usd > 6.0:
            raise ValueError("Vast job budget must be positive and cannot exceed USD 6")
        expected_thresholds = {
            "TRIỆU_CHỨNG",
            "TÊN_XÉT_NGHIỆM",
            "KẾT_QUẢ_XÉT_NGHIỆM",
            "CHẨN_ĐOÁN",
            "THUỐC",
        }
        if set(self.thresholds) != expected_thresholds:
            raise ValueError("Qwen thresholds must cover exactly the five Phase 1 labels")
        if any(not 0.0 <= value <= 1.0 for value in self.thresholds.values()):
            raise ValueError("Qwen thresholds must be between zero and one")

    @property
    def prefetch_command(self) -> tuple[str, ...]:
        """Return the explicit networked checkpoint acquisition command."""

        return (
            "hf",
            "download",
            self.model.model_id,
            "--revision",
            self.model.revision,
        )

    def verify_dataset_inputs(self) -> None:
        """Fail execution when reproducible instruction artifacts have not been built."""

        if not self.dataset_path.is_file() or not self.dataset_manifest_path.is_file():
            raise ValueError(
                "Qwen instruction dataset is absent; run the qwen-data builder first"
            )

    def verify_adapter_inputs(self) -> dict[str, Any] | None:
        """Verify an optional local adapter before model initialization."""

        if self.adapter is None:
            return None
        return self.adapter.verify(self.model)

    def relative_path(self, path: Path) -> str:
        """Project one resolved path into the portable run-root namespace."""

        try:
            return path.resolve().relative_to(self.run_root).as_posix()
        except ValueError as error:
            raise ValueError(f"Qwen run path escapes run_root: {path}") from error

    def to_dict(self) -> dict[str, Any]:
        """Serialize every behavior-bearing run parameter."""

        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model": self.model.to_dict(),
            "model_source_url": self.model_source_url,
            "model_license": self.model_license,
            "budget": self.budget.to_dict(),
            "dataset": {
                "path": self.relative_path(self.dataset_path),
                "manifest": self.relative_path(self.dataset_manifest_path),
            },
            "runtime": {
                "device": self.device,
                "dtype": self.dtype,
                "local_files_only": self.local_files_only,
                "max_window_characters": self.max_window_characters,
                "window_overlap_characters": self.window_overlap_characters,
                "structured_retries": self.structured_retries,
            },
            "generation": {
                "recall": _generation_dict(self.recall_generation),
                "targeted": _generation_dict(self.targeted_generation),
                "adjudication": _generation_dict(self.adjudication_generation),
            },
            "thresholds": dict(sorted(self.thresholds.items())),
            "remote": {"maximum_vast_cost_usd": self.maximum_vast_cost_usd},
        }
        # MODEL: omitting the optional key preserves the identity of pre-PEFT base-only specs.
        if self.adapter is not None:
            payload["adapter"] = self.adapter.to_dict(self)
        return payload


def load_phase1_qwen_run_spec(path: str | Path) -> Phase1QwenRunSpec:
    """Load a strict YAML run spec without importing Torch or Transformers."""

    config_path = Path(path).resolve()
    raw = read_yaml(config_path)
    if raw.get("schema_version") != "phase1-qwen-run.v1":
        raise ValueError("Unsupported Phase 1 Qwen run schema")
    run_root = _resolve(config_path.parent, _required_string(raw, "run_root"))
    try:
        config_path.relative_to(run_root)
    except ValueError as error:
        raise ValueError("Qwen config must live below run_root") from error
    model_raw = _mapping(raw, "model")
    budget_raw = _mapping(raw, "budget")
    model = _model_entry(model_raw)
    adapter = _adapter_spec(raw.get("adapter"), run_root)
    auxiliary_raw = budget_raw.get("auxiliary", [])
    if not isinstance(auxiliary_raw, list):
        raise ValueError("budget.auxiliary must be a list")
    auxiliary = tuple(_model_entry(_as_mapping(value, "auxiliary model")) for value in auxiliary_raw)
    budget = InferenceBudgetManifest(
        entries=(
            model,
            *((adapter.model,) if adapter is not None else ()),
            *auxiliary,
        ),
        maximum_parameters=int(budget_raw.get("maximum_parameters", 9_000_000_000)),
    )
    dataset = _mapping(raw, "dataset")
    runtime = _mapping(raw, "runtime")
    generation = _mapping(raw, "generation")
    thresholds_raw = _mapping(raw, "thresholds")
    remote = _mapping(raw, "remote")
    return Phase1QwenRunSpec(
        schema_version=str(raw["schema_version"]),
        run_id=_required_string(raw, "run_id"),
        config_path=config_path,
        run_root=run_root,
        model=model,
        model_source_url=_required_string(model_raw, "source_url"),
        model_license=_required_string(model_raw, "license"),
        adapter=adapter,
        budget=budget,
        dataset_path=_resolve(run_root, _required_string(dataset, "path")),
        dataset_manifest_path=_resolve(run_root, _required_string(dataset, "manifest")),
        device=str(runtime.get("device", "cuda")),
        dtype=str(runtime.get("dtype", "bf16")),
        local_files_only=bool(runtime.get("local_files_only", True)),
        max_window_characters=int(runtime.get("max_window_characters", 12_000)),
        window_overlap_characters=int(runtime.get("window_overlap_characters", 800)),
        structured_retries=int(runtime.get("structured_retries", 1)),
        recall_generation=_generation_config(_mapping(generation, "recall")),
        targeted_generation=_generation_config(_mapping(generation, "targeted")),
        adjudication_generation=_generation_config(_mapping(generation, "adjudication")),
        thresholds={str(key): float(value) for key, value in thresholds_raw.items()},
        maximum_vast_cost_usd=float(remote.get("maximum_vast_cost_usd", 6.0)),
    )


def _model_entry(raw: dict[str, Any]) -> ModelBudgetEntry:
    revision = _required_string(raw, "revision")
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise ValueError("Qwen model revision must be a full 40-character commit SHA")
    roles_raw = raw.get("roles")
    if not isinstance(roles_raw, list):
        raise ValueError("Model roles must be a list")
    return ModelBudgetEntry(
        artifact_id=_required_string(raw, "artifact_id"),
        model_id=_required_string(raw, "model_id"),
        revision=revision,
        parameter_count=int(raw["parameter_count"]),
        kind=str(raw.get("kind", "base")),  # type: ignore[arg-type]
        roles=tuple(sorted(str(value) for value in roles_raw)),
    )


def _adapter_spec(
    value: object,
    run_root: Path,
) -> Phase1PeftAdapterSpec | None:
    if value is None:
        return None
    raw = _as_mapping(value, "adapter")
    provenance = _mapping(raw, "provenance")
    return Phase1PeftAdapterSpec(
        model=_model_entry(raw),
        path=_resolve(run_root, _required_string(raw, "path")),
        fingerprint=_required_string(raw, "fingerprint"),
        provenance_manifest_path=_resolve(
            run_root,
            _required_string(provenance, "manifest"),
        ),
        provenance_manifest_sha256=_required_string(
            provenance,
            "manifest_sha256",
        ),
    )


def _generation_config(raw: dict[str, Any]) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=int(raw.get("max_new_tokens", 2048)),
        temperature=float(raw.get("temperature", 0.0)),
        top_p=float(raw.get("top_p", 1.0)),
        seed=int(raw.get("seed", 42)),
        enable_thinking=bool(raw.get("enable_thinking", False)),
    )


def _generation_dict(config: GenerationConfig) -> dict[str, Any]:
    return {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
        "enable_thinking": config.enable_thinking,
    }


def _mapping(raw: dict[str, Any], field: str) -> dict[str, Any]:
    return _as_mapping(raw.get(field), field)


def _as_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required_string(raw: MappingLike, field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _json_mapping(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not valid JSON: {path}") from error
    return _as_mapping(value, field)


MappingLike = dict[str, Any]
