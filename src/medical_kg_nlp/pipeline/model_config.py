"""Configuration records for optional local model adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from medical_kg_nlp.adapters.huggingface import HuggingFaceModelConfig
from medical_kg_nlp.schema.types import EntityType

__all__ = ["PipelineModelConfig"]


@dataclass(frozen=True)
class PipelineModelConfig:
    """Optional model stages assembled by :class:`PipelineFactory`."""

    entity_extractor: HuggingFaceModelConfig | None = None
    entity_label_map: tuple[tuple[str, EntityType], ...] = ()
    entity_stride: int = 64
    entity_combine_with_dictionary: bool = False
    candidate_reranker: HuggingFaceModelConfig | None = None
    candidate_reranker_weight: float = 0.75
    candidate_positive_label_index: int = 1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PipelineModelConfig":
        """Parse configured model stages while requiring pinned revisions."""

        entity_payload = _optional_mapping(payload.get("entity_extractor"), "entity_extractor")
        reranker_payload = _optional_mapping(
            payload.get("candidate_reranker"),
            "candidate_reranker",
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
        entity_label_map = _label_map(entity_payload or {})
        entity_stride = _integer(entity_payload or {}, "stride", cls.entity_stride)
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
            entity_combine_with_dictionary=combine_with_dictionary,
            candidate_reranker=reranker_model,
            candidate_reranker_weight=reranker_weight,
            candidate_positive_label_index=positive_label_index,
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
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"models.{key} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"models.{key} must be between 0 and 1")
    return result
