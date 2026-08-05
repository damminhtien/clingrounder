"""Pinned local cross-encoder for bounded candidate reranking."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from medical_kg_nlp.adapters.huggingface.config import HuggingFaceModelConfig
from medical_kg_nlp.adapters.huggingface.runtime import (
    _load_runtime,
    _move_model_inputs,
    _probability,
)
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.batch import CandidateRerankRequest

if TYPE_CHECKING:
    from medical_kg_nlp.pipeline.ports import CandidateRerankerPort

__all__ = ["HuggingFaceCrossEncoderAdapter"]


class HuggingFaceCrossEncoderAdapter:
    """Rerank a bounded candidate list with a pinned sequence-classification model."""

    def __init__(
        self,
        config: HuggingFaceModelConfig,
        *,
        model_weight: float = 0.75,
        positive_label_index: int = 1,
        base_reranker: CandidateRerankerPort | None = None,
        max_pairs_per_batch: int | None = None,
        max_tokens: int | None = None,
        cancellation_hook: Callable[[], bool] | None = None,
    ) -> None:
        if not 0.0 <= model_weight <= 1.0:
            raise ValueError("model_weight must be between 0 and 1")
        if positive_label_index < 0:
            raise ValueError("positive_label_index must be non-negative")
        self.config = config
        self.model_weight = model_weight
        self.positive_label_index = positive_label_index
        self.base_reranker = base_reranker
        self.max_pairs_per_batch = max_pairs_per_batch
        self.max_tokens = max_tokens
        self.cancellation_hook = cancellation_hook
        if max_pairs_per_batch is not None and max_pairs_per_batch < 1:
            raise ValueError("max_pairs_per_batch must be at least 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self._loaded: tuple[Any, Any, Any] | None = None
        self._tokenizer_calls = 0
        self._model_forward_passes = 0

    def rerank(
        self,
        candidates: list[Candidate],
        context_window: str = "",
        mention: str = "",
    ) -> list[Candidate]:
        """Score one request through the same path used by the batch API."""

        request = CandidateRerankRequest(
            entity_id="__scalar__",
            mention=mention,
            context_window=context_window,
            candidates=tuple(candidates),
        )
        return self.rerank_batch((request,))[request.entity_id]

    def rerank_batch(
        self,
        requests: tuple[CandidateRerankRequest, ...],
    ) -> dict[str, list[Candidate]]:
        """Rerank independent entity lists with one flattened, bounded model workload."""

        entity_ids = [request.entity_id for request in requests]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Candidate rerank batch contains duplicate entity IDs")
        prepared: dict[str, list[Candidate]] = {}
        pairs: list[tuple[str, str]] = []
        pair_owners: list[str] = []
        for request in requests:
            base_candidates = self._prepare_candidates(request)
            prepared[request.entity_id] = base_candidates
            query = (
                request.mention
                if not request.context_window
                else f"{request.mention}\n{request.context_window}"
            )
            for candidate in base_candidates:
                pairs.append((query, candidate.canonical_name))
                pair_owners.append(request.entity_id)
        scores = self._score_pairs(pairs) if pairs else []
        grouped_scores: dict[str, list[float]] = {entity_id: [] for entity_id in entity_ids}
        for entity_id, score in zip(pair_owners, scores, strict=True):
            grouped_scores[entity_id].append(score)
        return {
            entity_id: self._apply_scores(
                prepared[entity_id], grouped_scores[entity_id]
            )
            for entity_id in entity_ids
        }

    def stats(self) -> dict[str, int]:
        """Return counters useful for measured batch-vs-scalar benchmarks."""

        return {
            "tokenizer_calls": self._tokenizer_calls,
            "model_forward_passes": self._model_forward_passes,
        }

    def close(self) -> None:
        """Release model references and move a loaded model off accelerator memory."""

        loaded = self._loaded
        if loaded is None:
            return
        model = loaded[2]
        if callable(getattr(model, "cpu", None)):
            model.cpu()
        self._loaded = None

    def _prepare_candidates(self, request: CandidateRerankRequest) -> list[Candidate]:
        if self.base_reranker is None:
            return list(request.candidates)
        return self.base_reranker.rerank(
            list(request.candidates),
            context_window=request.context_window,
            mention=request.mention,
        )

    def _apply_scores(
        self,
        candidates: list[Candidate],
        scores: list[float],
    ) -> list[Candidate]:
        if len(candidates) != len(scores):
            raise ValueError("Cross-encoder score count does not match candidate count")
        # INVARIANT: candidate identity, evidence, and qualification metadata are copied by
        # dataclasses.replace; only the numeric score changes before deterministic sorting.
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
        for batch in self._bounded_pair_batches(pairs):
            if self.cancellation_hook is not None and self.cancellation_hook():
                raise TimeoutError("Cross-encoder reranking cancelled by cancellation_hook")
            encoded = tokenizer(
                [left for left, _ in batch],
                [right for _, right in batch],
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            self._tokenizer_calls += 1
            model_inputs = _move_model_inputs(encoded, self.config.device)
            with torch.inference_mode():
                logits = model(**model_inputs).logits
                self._model_forward_passes += 1
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

    def _bounded_pair_batches(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[list[tuple[str, str]]]:
        """Bound pair count and a conservative token estimate before tokenization."""

        pair_limit = self.max_pairs_per_batch or self.config.batch_size
        token_limit = self.max_tokens
        batches: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []
        estimated_tokens = 0
        for pair in pairs:
            pair_tokens = _estimate_pair_tokens(pair)
            if current and (
                len(current) >= pair_limit
                or (token_limit is not None and estimated_tokens + pair_tokens > token_limit)
            ):
                batches.append(current)
                current = []
                estimated_tokens = 0
            current.append(pair)
            estimated_tokens += pair_tokens
        if current:
            batches.append(current)
        return batches

    def _runtime(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            self._loaded = _load_runtime(
                self.config,
                auto_model_class="AutoModelForSequenceClassification",
            )
        return self._loaded


def _estimate_pair_tokens(pair: tuple[str, str]) -> int:
    """Conservative, tokenizer-independent guard used before model tokenization."""

    return max(1, len(pair[0].split()) + len(pair[1].split()) + 3)
