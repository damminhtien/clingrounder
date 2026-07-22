"""Local token-classification NER with raw offset projection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from medical_kg_nlp.adapters.huggingface.config import HuggingFaceModelConfig
from medical_kg_nlp.adapters.huggingface.runtime import (
    _as_nested_list,
    _load_runtime,
    _probability,
    _slice_model_inputs,
)
from medical_kg_nlp.adapters.model_spans import TokenPrediction, project_bio_predictions
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["HuggingFaceTokenClassifierAdapter"]


class HuggingFaceTokenClassifierAdapter:
    """Run local token classification and project fast-tokenizer offsets to raw text."""

    def __init__(
        self,
        config: HuggingFaceModelConfig,
        *,
        label_map: Mapping[str, EntityType] | None = None,
        stride: int = 64,
        confidence_thresholds: Mapping[EntityType, float] | None = None,
        default_confidence_threshold: float = 0.0,
    ) -> None:
        if stride < 0 or stride >= config.max_length - 2:
            raise ValueError("stride must fit inside max_length")
        self.config = config
        self.label_map = dict(label_map or {})
        self.stride = stride
        self.confidence_thresholds = dict(confidence_thresholds or {})
        self.default_confidence_threshold = default_confidence_threshold
        self._loaded: tuple[Any, Any, Any] | None = None

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        """Extract entities while preserving exact source-text slices."""

        if not source_text:
            return []
        torch, tokenizer, model = self._runtime()
        encoded = tokenizer(
            source_text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            stride=self.stride,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        offsets = _as_nested_list(encoded.pop("offset_mapping"))
        encoded.pop("overflow_to_sample_mapping", None)
        id2label = getattr(model.config, "id2label", {})
        predictions: list[TokenPrediction] = []
        window_count = len(offsets)
        for batch_start in range(0, window_count, self.config.batch_size):
            batch_end = min(window_count, batch_start + self.config.batch_size)
            model_inputs = _slice_model_inputs(
                encoded,
                batch_start,
                batch_end,
                self.config.device,
            )
            with torch.inference_mode():
                logits = model(**model_inputs).logits
                probabilities = torch.softmax(logits, dim=-1)
                scores, label_ids = probabilities.max(dim=-1)
            score_rows = _as_nested_list(scores)
            label_rows = _as_nested_list(label_ids)
            for row_index, (score_row, label_row) in enumerate(
                zip(score_rows, label_rows, strict=True),
                start=batch_start,
            ):
                for raw_offset, raw_label_id, raw_score in zip(
                    offsets[row_index],
                    label_row,
                    score_row,
                    strict=True,
                ):
                    start, end = int(raw_offset[0]), int(raw_offset[1])
                    if start == end:
                        continue
                    label_id = int(raw_label_id)
                    label = str(id2label.get(label_id, id2label.get(str(label_id), label_id)))
                    predictions.append(
                        TokenPrediction(
                            start=start,
                            end=end,
                            label=label,
                            score=float(raw_score),
                        )
                    )

        projected = project_bio_predictions(
            source_text,
            predictions,
            label_map=self.label_map,
            confidence_thresholds=self.confidence_thresholds,
            default_confidence_threshold=self.default_confidence_threshold,
        )
        entities: list[EntityAnnotation] = []
        for index, item in enumerate(projected, start=1):
            start, end = item.span
            text = source_text[start:end]
            entity = EntityAnnotation(
                id=f"M{index:04d}",
                span=item.span,
                text=text,
                normalized_text=normalize_for_match(text),
                type=item.entity_type,
                confidence=_probability(item.confidence),
            )
            # MODEL: fast-tokenizer offsets are accepted only after raw-slice validation.
            entity.validate_offsets(source_text)
            entities.append(entity)
        return entities

    def _runtime(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            self._loaded = _load_runtime(
                self.config,
                auto_model_class="AutoModelForTokenClassification",
                require_fast_tokenizer=True,
            )
        return self._loaded
