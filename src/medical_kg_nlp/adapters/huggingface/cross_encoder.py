"""Pinned local cross-encoder for bounded candidate reranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from medical_kg_nlp.adapters.huggingface.config import HuggingFaceModelConfig
from medical_kg_nlp.adapters.huggingface.runtime import (
    _load_runtime,
    _move_model_inputs,
    _probability,
)
from medical_kg_nlp.linking.candidate import Candidate

__all__ = ["HuggingFaceCrossEncoderAdapter"]


class HuggingFaceCrossEncoderAdapter:
    """Rerank a bounded candidate list with a pinned sequence-classification model."""

    def __init__(
        self,
        config: HuggingFaceModelConfig,
        *,
        model_weight: float = 0.75,
        positive_label_index: int = 1,
    ) -> None:
        if not 0.0 <= model_weight <= 1.0:
            raise ValueError("model_weight must be between 0 and 1")
        if positive_label_index < 0:
            raise ValueError("positive_label_index must be non-negative")
        self.config = config
        self.model_weight = model_weight
        self.positive_label_index = positive_label_index
        self._loaded: tuple[Any, Any, Any] | None = None

    def rerank(
        self,
        candidates: list[Candidate],
        context_window: str = "",
        mention: str = "",
    ) -> list[Candidate]:
        """Score mention/concept pairs in bounded batches and preserve candidate identity."""

        if not candidates:
            return []
        query = mention if not context_window else f"{mention}\n{context_window}"
        scores = self._score_pairs(
            [(query, candidate.canonical_name) for candidate in candidates]
        )
        reranked = [
            replace(
                candidate,
                score=_probability(
                    (1.0 - self.model_weight) * candidate.score
                    + self.model_weight * model_score
                ),
            )
            for candidate, model_score in zip(candidates, scores, strict=True)
        ]
        return sorted(
            reranked,
            key=lambda candidate: (
                -candidate.score,
                candidate.code_system.value,
                candidate.code or "",
                candidate.concept_id,
            ),
        )

    def _score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        torch, tokenizer, model = self._runtime()
        output: list[float] = []
        for start in range(0, len(pairs), self.config.batch_size):
            batch = pairs[start : start + self.config.batch_size]
            encoded = tokenizer(
                [left for left, _ in batch],
                [right for _, right in batch],
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            model_inputs = _move_model_inputs(encoded, self.config.device)
            with torch.inference_mode():
                logits = model(**model_inputs).logits
                if int(logits.shape[-1]) == 1:
                    probabilities = torch.sigmoid(logits.squeeze(-1))
                else:
                    if self.positive_label_index >= int(logits.shape[-1]):
                        raise ValueError("positive_label_index exceeds model label count")
                    probabilities = torch.softmax(logits, dim=-1)[
                        :, self.positive_label_index
                    ]
            output.extend(float(value) for value in probabilities.detach().cpu().tolist())
        return output

    def _runtime(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            self._loaded = _load_runtime(
                self.config,
                auto_model_class="AutoModelForSequenceClassification",
            )
        return self._loaded
