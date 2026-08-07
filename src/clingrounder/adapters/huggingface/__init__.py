"""Lazy, local-only Hugging Face adapters for replaceable model stages."""

from clingrounder.adapters.huggingface.config import HuggingFaceModelConfig
from clingrounder.adapters.huggingface.cross_encoder import (
    HuggingFaceCrossEncoderAdapter,
)
from clingrounder.adapters.huggingface.multiclass_text_classifier import (
    HuggingFaceMulticlassTextClassifierAdapter,
)
from clingrounder.adapters.huggingface.runtime import OptionalModelDependencyError
from clingrounder.adapters.huggingface.source_token_classifier import (
    HuggingFaceSourceTokenClassifierAdapter,
)
from clingrounder.adapters.huggingface.text_encoder import HuggingFaceTextEncoderAdapter
from clingrounder.adapters.huggingface.token_classifier import (
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
