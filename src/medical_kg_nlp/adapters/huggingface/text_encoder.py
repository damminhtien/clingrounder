"""Pinned local text encoder for dense candidate retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from medical_kg_nlp.adapters.huggingface.config import HuggingFaceModelConfig
from medical_kg_nlp.adapters.huggingface.runtime import _load_runtime, _move_model_inputs

__all__ = ["HuggingFaceTextEncoderAdapter"]


class HuggingFaceTextEncoderAdapter:
    """Encode text batches into normalized dense vectors using a pinned local model."""

    def __init__(self, config: HuggingFaceModelConfig) -> None:
        self.config = config
        self._loaded: tuple[Any, Any, Any] | None = None

    def encode(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Mean-pool attention-masked hidden states and L2-normalize each vector."""

        if not texts:
            return []
        torch, tokenizer, model = self._runtime()
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = list(texts[start : start + self.config.batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            model_inputs = _move_model_inputs(encoded, self.config.device)
            with torch.inference_mode():
                hidden = model(**model_inputs).last_hidden_state
                mask = model_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(
                tuple(float(value) for value in row)
                for row in normalized.detach().cpu().tolist()
            )
        return vectors

    def _runtime(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            self._loaded = _load_runtime(
                self.config,
                auto_model_class="AutoModel",
            )
        return self._loaded
