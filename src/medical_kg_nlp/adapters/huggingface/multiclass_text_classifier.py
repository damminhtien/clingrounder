"""Pinned local multi-class text classification for benchmark-owned verifier adapters."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from medical_kg_nlp.adapters.huggingface.config import HuggingFaceModelConfig
from medical_kg_nlp.adapters.huggingface.runtime import (
    _as_nested_list,
    _load_runtime,
    _move_model_inputs,
)

__all__ = ["HuggingFaceMulticlassTextClassifierAdapter"]


class HuggingFaceMulticlassTextClassifierAdapter:
    """Run a local sequence classifier and expose stable label-keyed probabilities.

    This adapter owns only model execution. Task code owns the input contract, labels, thresholds,
    and output decisions, which keeps the Hugging Face dependency reusable beyond Phase 1.
    """

    def __init__(
        self,
        config: HuggingFaceModelConfig,
        *,
        labels: Sequence[str],
    ) -> None:
        canonical_labels = tuple(labels)
        if not canonical_labels or len(set(canonical_labels)) != len(canonical_labels):
            raise ValueError("Multiclass classifier labels must be non-empty and unique")
        if any(not label.strip() for label in canonical_labels):
            raise ValueError("Multiclass classifier labels must be non-empty strings")
        self.config = config
        self.labels = canonical_labels
        self._loaded: tuple[Any, Any, Any] | None = None

    @property
    def provenance(self) -> str:
        """Return the immutable model identity recorded in downstream traces."""

        return self.config.provenance

    def predict(self, texts: Sequence[str]) -> tuple[tuple[tuple[str, float], ...], ...]:
        """Return one normalized label distribution for every non-empty rendered input."""

        if not texts:
            return ()
        if any(not text.strip() for text in texts):
            raise ValueError("Multiclass classifier inputs must be non-empty")
        rows = self._score_texts(texts)
        if len(rows) != len(texts):
            raise ValueError("Multiclass model output count does not match input count")
        return tuple(
            tuple(zip(self.labels, _normalize_distribution(row), strict=True))
            for row in rows
        )

    def _score_texts(self, texts: Sequence[str]) -> list[list[float]]:
        torch, tokenizer, model = self._runtime()
        label_index = _model_label_index(model, self.labels)
        output: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            encoded = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            model_inputs = _move_model_inputs(encoded, self.config.device)
            with torch.inference_mode():
                probabilities = torch.softmax(model(**model_inputs).logits, dim=-1)
            for raw_row in _as_nested_list(probabilities):
                if not isinstance(raw_row, list):
                    raise ValueError("Multiclass model probability row is malformed")
                output.append([float(raw_row[index]) for index in label_index])
        return output

    def _runtime(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            self._loaded = _load_runtime(
                self.config,
                auto_model_class="AutoModelForSequenceClassification",
            )
        return self._loaded


def _model_label_index(model: Any, labels: Sequence[str]) -> tuple[int, ...]:
    """Resolve checkpoint label names once and reject a classifier with another contract."""

    raw_id_to_label = getattr(getattr(model, "config", None), "id2label", {})
    if not isinstance(raw_id_to_label, dict):
        raise ValueError("Multiclass classifier checkpoint has no id2label mapping")
    normalized = {str(value): int(key) for key, value in raw_id_to_label.items()}
    if set(normalized) != set(labels):
        raise ValueError("Multiclass classifier checkpoint labels do not match the task contract")
    return tuple(normalized[label] for label in labels)


def _normalize_distribution(values: Sequence[float]) -> tuple[float, ...]:
    """Validate model output before it becomes a calibrated decision input."""

    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Multiclass classifier probabilities must be finite and non-negative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("Multiclass classifier probabilities must have positive mass")
    return tuple(value / total for value in values)
