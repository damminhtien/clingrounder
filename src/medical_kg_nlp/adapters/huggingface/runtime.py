"""Framework-loading and tensor helpers kept out of core adapter imports."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from medical_kg_nlp.adapters.huggingface.config import HuggingFaceModelConfig

__all__ = ["OptionalModelDependencyError"]


class OptionalModelDependencyError(RuntimeError):
    """Explain how to enable model adapters without affecting the core install."""


def _load_runtime(
    config: HuggingFaceModelConfig,
    *,
    auto_model_class: str,
    require_fast_tokenizer: bool = False,
) -> tuple[Any, Any, Any]:
    torch, transformers = _import_model_dependencies()
    source_options = (
        {} if config.subfolder is None else {"subfolder": config.subfolder}
    )
    # MODEL: local_files_only prevents a configured pipeline from becoming a hosted dependency.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
        **source_options,
    )
    if require_fast_tokenizer and not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("Token-classification NER requires a fast tokenizer offset mapping")
    model_class = getattr(transformers, auto_model_class)
    model = model_class.from_pretrained(
        config.model_id,
        revision=config.revision,
        local_files_only=True,
        trust_remote_code=False,
        **source_options,
    )
    model.to(config.device)
    model.eval()
    return torch, tokenizer, model


def _import_model_dependencies() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalModelDependencyError(
            "Local Hugging Face adapters require the 'ml' extra: uv sync --extra ml"
        ) from error
    return torch, transformers


def _slice_model_inputs(
    encoded: Mapping[str, Any],
    start: int,
    end: int,
    device: str,
) -> dict[str, Any]:
    return {
        key: value[start:end].to(device)
        for key, value in encoded.items()
        if hasattr(value, "to")
    }


def _move_model_inputs(encoded: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if hasattr(value, "to")
    }


def _as_nested_list(value: Any) -> list[Any]:
    detached = value.detach() if hasattr(value, "detach") else value
    cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
    converted = cpu_value.tolist() if hasattr(cpu_value, "tolist") else cpu_value
    if not isinstance(converted, list):
        raise TypeError("Expected a tensor-like value convertible to a list")
    return converted


def _probability(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Model probability must be finite")
    return min(1.0, max(0.0, value))
