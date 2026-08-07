"""Source-taxonomy token classification for verifier and transfer-learning paths."""

from __future__ import annotations

from collections.abc import Mapping

from clingrounder.adapters.huggingface.config import HuggingFaceModelConfig
from clingrounder.adapters.huggingface.token_classifier import (
    HuggingFaceTokenClassifierAdapter,
)
from clingrounder.adapters.model_spans import (
    ProjectedSourceEntity,
    project_source_bio_predictions,
)

__all__ = ["HuggingFaceSourceTokenClassifierAdapter"]


class HuggingFaceSourceTokenClassifierAdapter:
    """Preserve checkpoint labels instead of coercing them into core entity types."""

    def __init__(
        self,
        config: HuggingFaceModelConfig,
        *,
        stride: int = 64,
        confidence_thresholds: Mapping[str, float] | None = None,
        default_confidence_threshold: float = 0.0,
    ) -> None:
        self.config = config
        self._token_classifier = HuggingFaceTokenClassifierAdapter(
            config,
            stride=stride,
        )
        self.confidence_thresholds = dict(confidence_thresholds or {})
        self.default_confidence_threshold = default_confidence_threshold

    def extract(self, source_text: str) -> list[ProjectedSourceEntity]:
        """Project raw checkpoint BIO labels onto exact source-text slices."""

        predictions = self._token_classifier.predict_token_labels(source_text)
        entities = project_source_bio_predictions(
            source_text,
            predictions,
            confidence_thresholds=self.confidence_thresholds,
            default_confidence_threshold=self.default_confidence_threshold,
        )
        # INVARIANT: broad verifier labels still target the immutable source text.
        for entity in entities:
            start, end = entity.span
            if source_text[start:end] == "":
                raise ValueError("Source token classifier emitted an empty raw span")
        return entities

    def close(self) -> None:
        """Delegate cleanup to the wrapped token classifier."""

        self._token_classifier.close()
