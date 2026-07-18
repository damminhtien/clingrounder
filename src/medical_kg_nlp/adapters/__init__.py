"""Built-in adapters for pipeline ports."""

from medical_kg_nlp.adapters.huggingface import (
    HuggingFaceCrossEncoderAdapter,
    HuggingFaceModelConfig,
    HuggingFaceTextEncoderAdapter,
    HuggingFaceTokenClassifierAdapter,
    OptionalModelDependencyError,
)
from medical_kg_nlp.adapters.hybrid import HybridEntityExtractorAdapter
from medical_kg_nlp.adapters.rules import (
    DictionaryCandidateAdapter,
    KGValidatorAdapter,
    RuleAssertionClassifierAdapter,
    RuleEntityExtractorAdapter,
    RuleRelationExtractorAdapter,
)

__all__ = [
    "DictionaryCandidateAdapter",
    "HuggingFaceCrossEncoderAdapter",
    "HuggingFaceModelConfig",
    "HuggingFaceTextEncoderAdapter",
    "HuggingFaceTokenClassifierAdapter",
    "HybridEntityExtractorAdapter",
    "KGValidatorAdapter",
    "OptionalModelDependencyError",
    "RuleAssertionClassifierAdapter",
    "RuleEntityExtractorAdapter",
    "RuleRelationExtractorAdapter",
]
