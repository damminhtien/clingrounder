"""Pinned Qwen benchmark specifications with local model and Vast cost gates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.adapters.generative import (
    GenerationConfig,
    InferenceBudgetManifest,
    ModelBudgetEntry,
)
from medical_kg_nlp.utils.io import read_yaml

__all__ = ["Phase1QwenRunSpec", "load_phase1_qwen_run_spec"]

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


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
        # INVARIANT: specs may be inspected before derived data is built, but paths cannot escape
        # the portable run root. Execution calls ``verify_dataset_inputs`` explicitly.
        self.relative_path(self.dataset_path)
        self.relative_path(self.dataset_manifest_path)
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

    def relative_path(self, path: Path) -> str:
        """Project one resolved path into the portable run-root namespace."""

        try:
            return path.resolve().relative_to(self.run_root).as_posix()
        except ValueError as error:
            raise ValueError(f"Qwen run path escapes run_root: {path}") from error

    def to_dict(self) -> dict[str, Any]:
        """Serialize every behavior-bearing run parameter."""

        return {
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
    auxiliary_raw = budget_raw.get("auxiliary", [])
    if not isinstance(auxiliary_raw, list):
        raise ValueError("budget.auxiliary must be a list")
    auxiliary = tuple(_model_entry(_as_mapping(value, "auxiliary model")) for value in auxiliary_raw)
    budget = InferenceBudgetManifest(
        entries=(model, *auxiliary),
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


MappingLike = dict[str, Any]
