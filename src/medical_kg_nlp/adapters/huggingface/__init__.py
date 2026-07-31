"""Lazy, local-only Hugging Face adapters for replaceable model stages."""

from medical_kg_nlp.adapters.huggingface.config import HuggingFaceModelConfig
from medical_kg_nlp.adapters.huggingface.cross_encoder import (
    HuggingFaceCrossEncoderAdapter,
)
from medical_kg_nlp.adapters.huggingface.multiclass_text_classifier import (
    HuggingFaceMulticlassTextClassifierAdapter,
)
from medical_kg_nlp.adapters.huggingface.runtime import OptionalModelDependencyError
from medical_kg_nlp.adapters.huggingface.source_token_classifier import (
    HuggingFaceSourceTokenClassifierAdapter,
)
from medical_kg_nlp.adapters.huggingface.text_encoder import HuggingFaceTextEncoderAdapter
from medical_kg_nlp.adapters.huggingface.token_classifier import (
    HuggingFaceTokenClassifierAdapter,
)

__all__ = [
    "HuggingFaceCrossEncoderAdapter",
    "HuggingFaceModelConfig",
    "HuggingFaceMulticlassTextClassifierAdapter",
    "HuggingFaceSourceTokenClassifierAdapter",
    "HuggingFaceTextEncoderAdapter",
    "HuggingFaceTokenClassifierAdapter",
    "OptionalModelDependencyError",
]
