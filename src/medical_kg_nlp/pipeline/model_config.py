"""Configuration records for optional local model adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from medical_kg_nlp.adapters.generative import GenerationConfig
from medical_kg_nlp.adapters.huggingface import HuggingFaceModelConfig
from medical_kg_nlp.schema.types import EntityType

__all__ = ["ListwiseRerankerModelConfig", "PipelineModelConfig"]


@dataclass(frozen=True)
class ListwiseRerankerModelConfig:
    """Pinned local causal-LM and bounded listwise inference policy."""

    model: HuggingFaceModelConfig
    generation: GenerationConfig
    dtype: Literal["auto", "bf16", "fp16", "fp32"] = "bf16"
    local_files_only: bool = True
    candidate_limit: int = 12
    model_weight: float = 0.75
    shuffle_seed: int = 42
    structured_retries: int = 1

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> "ListwiseRerankerModelConfig":
        model = HuggingFaceModelConfig.from_mapping(
            payload,
            name="candidate_listwise_reranker",
        )
        dtype = payload.get("dtype", cls.dtype)
        if dtype not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError(
                "models.candidate_listwise_reranker.dtype must be auto, bf16, fp16, or fp32"
            )
        local_files_only = payload.get("local_files_only", cls.local_files_only)
        if not isinstance(local_files_only, bool):
            raise ValueError(
                "models.candidate_listwise_reranker.local_files_only must be boolean"
            )
        candidate_limit = _integer(payload, "candidate_limit", cls.candidate_limit)
        if not 2 <= candidate_limit <= 12:
            raise ValueError(
                "models.candidate_listwise_reranker.candidate_limit must be between 2 and 12"
            )
        model_weight = _probability(
            payload,
            "model_weight",
            cls.model_weight,
        )
        shuffle_seed = _integer(payload, "shuffle_seed", cls.shuffle_seed)
        structured_retries = _integer(
            payload,
            "structured_retries",
            cls.structured_retries,
        )
        if structured_retries < 0:
            raise ValueError(
                "models.candidate_listwise_reranker.structured_retries cannot be negative"
            )
        max_new_tokens = _integer(
            payload,
            "max_new_tokens",
            512,
        )
        temperature = _nonnegative_float(payload, "temperature", 0.0)
        top_p = _probability_value(
            payload.get("top_p", 1.0),
            "models.candidate_listwise_reranker.top_p",
        )
        seed = _integer(payload, "seed", 42)
        enable_thinking = payload.get("enable_thinking", False)
        if not isinstance(enable_thinking, bool):
            raise ValueError(
                "models.candidate_listwise_reranker.enable_thinking must be boolean"
            )
        return cls(
            model=model,
            generation=GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                enable_thinking=enable_thinking,
                stop_on_complete_json=True,
            ),
            dtype=cast(Literal["auto", "bf16", "fp16", "fp32"], dtype),
            local_files_only=local_files_only,
            candidate_limit=candidate_limit,
            model_weight=model_weight,
            shuffle_seed=shuffle_seed,
            structured_retries=structured_retries,
        )


@dataclass(frozen=True)
class PipelineModelConfig:
    """Optional model stages assembled by :class:`PipelineFactory`."""

    entity_extractor: HuggingFaceModelConfig | None = None
    entity_label_map: tuple[tuple[str, EntityType], ...] = ()
    entity_stride: int = 64
    entity_default_confidence_threshold: float = 0.0
    entity_confidence_thresholds: tuple[tuple[EntityType, float], ...] = ()
    entity_combine_with_dictionary: bool = False
    candidate_reranker: HuggingFaceModelConfig | None = None
    candidate_reranker_weight: float = 0.75
    candidate_positive_label_index: int = 1
    candidate_listwise_reranker: ListwiseRerankerModelConfig | None = None
    candidate_dense_encoder: HuggingFaceModelConfig | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PipelineModelConfig":
        """Parse configured model stages while requiring pinned revisions."""

        _reject_unknown(payload, {
            "entity_extractor",
            "candidate_reranker",
            "candidate_listwise_reranker",
            "candidate_dense_encoder",
        }, "models")

        entity_payload = _optional_mapping(payload.get("entity_extractor"), "entity_extractor")
        reranker_payload = _optional_mapping(
            payload.get("candidate_reranker"),
            "candidate_reranker",
        )
        listwise_payload = _optional_mapping(
            payload.get("candidate_listwise_reranker"),
            "candidate_listwise_reranker",
        )
        dense_payload = _optional_mapping(
            payload.get("candidate_dense_encoder"),
            "candidate_dense_encoder",
        )
        _reject_unknown(
            entity_payload or {},
            _HF_KEYS
            | {
                "run_spec",
                "stride",
                "default_confidence_threshold",
                "confidence_thresholds",
                "label_map",
                "combine_with_dictionary",
            },
            "models.entity_extractor",
        )
        _reject_unknown(
            reranker_payload or {},
            _HF_KEYS | {"model_weight", "positive_label_index"},
            "models.candidate_reranker",
        )
        _reject_unknown(
            listwise_payload or {},
            _HF_KEYS
            | {
                "dtype",
                "local_files_only",
                "candidate_limit",
                "model_weight",
                "shuffle_seed",
                "structured_retries",
                "max_new_tokens",
                "temperature",
                "top_p",
                "seed",
                "enable_thinking",
            },
            "models.candidate_listwise_reranker",
        )
        _reject_unknown(dense_payload or {}, _HF_KEYS, "models.candidate_dense_encoder")
        if reranker_payload is not None and listwise_payload is not None:
            raise ValueError(
                "Configure only one of candidate_reranker or candidate_listwise_reranker"
            )
        entity_model = (
            HuggingFaceModelConfig.from_mapping(entity_payload, name="entity_extractor")
            if entity_payload is not None
            else None
        )
        reranker_model = (
            HuggingFaceModelConfig.from_mapping(
                reranker_payload,
                name="candidate_reranker",
            )
            if reranker_payload is not None
            else None
        )
        listwise_model = (
            ListwiseRerankerModelConfig.from_mapping(listwise_payload)
            if listwise_payload is not None
            else None
        )
        dense_model = (
            HuggingFaceModelConfig.from_mapping(
                dense_payload,
                name="candidate_dense_encoder",
            )
            if dense_payload is not None
            else None
        )
        entity_label_map = _label_map(entity_payload or {})
        entity_stride = _integer(entity_payload or {}, "stride", cls.entity_stride)
        default_confidence_threshold = _probability(
            entity_payload or {},
            "default_confidence_threshold",
            cls.entity_default_confidence_threshold,
        )
        confidence_thresholds = _confidence_thresholds(entity_payload or {})
        combine_with_dictionary = _boolean(
            entity_payload or {},
            "combine_with_dictionary",
            cls.entity_combine_with_dictionary,
        )
        reranker_weight = _probability(
            reranker_payload or {},
            "model_weight",
            cls.candidate_reranker_weight,
        )
        positive_label_index = _integer(
            reranker_payload or {},
            "positive_label_index",
            cls.candidate_positive_label_index,
        )
        if entity_stride < 0:
            raise ValueError("models.entity_extractor.stride must be non-negative")
        if positive_label_index < 0:
            raise ValueError(
                "models.candidate_reranker.positive_label_index must be non-negative"
            )
        return cls(
            entity_extractor=entity_model,
            entity_label_map=entity_label_map,
            entity_stride=entity_stride,
            entity_default_confidence_threshold=default_confidence_threshold,
            entity_confidence_thresholds=confidence_thresholds,
            entity_combine_with_dictionary=combine_with_dictionary,
            candidate_reranker=reranker_model,
            candidate_reranker_weight=reranker_weight,
            candidate_positive_label_index=positive_label_index,
            candidate_listwise_reranker=listwise_model,
            candidate_dense_encoder=dense_model,
        )


def _optional_mapping(
    value: object,
    name: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"models.{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _reject_unknown(
    payload: Mapping[str, object],
    allowed: set[str],
    path: str,
) -> None:
    unknown = sorted(str(key) for key in payload if str(key) not in allowed)
    if unknown:
        raise ValueError(f"Unknown pipeline config keys at {path}: {', '.join(unknown)}")


_HF_KEYS = {
    "model_id",
    "revision",
    "device",
    "batch_size",
    "max_length",
    "max_pairs_per_batch",
    "max_tokens",
    "subfolder",
}


def _label_map(payload: Mapping[str, object]) -> tuple[tuple[str, EntityType], ...]:
    raw = payload.get("label_map", {})
    if not isinstance(raw, Mapping):
        raise ValueError("models.entity_extractor.label_map must be a mapping")
    return tuple(
        sorted(
            (str(label), EntityType(str(entity_type)))
            for label, entity_type in raw.items()
        )
    )


def _confidence_thresholds(
    payload: Mapping[str, object],
) -> tuple[tuple[EntityType, float], ...]:
    raw = payload.get("confidence_thresholds", {})
    if not isinstance(raw, Mapping):
        raise ValueError("models.entity_extractor.confidence_thresholds must be a mapping")
    thresholds: list[tuple[EntityType, float]] = []
    for raw_entity_type, raw_threshold in raw.items():
        try:
            entity_type = EntityType(str(raw_entity_type).upper())
        except ValueError as exc:
            raise ValueError(
                "models.entity_extractor.confidence_thresholds has unknown entity type "
                f"{raw_entity_type!r}"
            ) from exc
        threshold = _probability_value(
            raw_threshold,
            f"models.entity_extractor.confidence_thresholds.{entity_type.value}",
        )
        thresholds.append((entity_type, threshold))
    return tuple(sorted(thresholds, key=lambda item: item[0].value))


def _integer(payload: Mapping[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"models.{key} must be an integer")
    return value


def _boolean(payload: Mapping[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"models.entity_extractor.{key} must be a boolean")
    return value


def _probability(payload: Mapping[str, object], key: str, default: float) -> float:
    return _probability_value(payload.get(key, default), f"models.{key}")


def _probability_value(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be between 0 and 1")
    return result


def _nonnegative_float(
    payload: Mapping[str, object],
    key: str,
    default: float,
) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"models.{key} must be numeric")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"models.{key} cannot be negative")
    return result
