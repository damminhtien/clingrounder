"""Built-in adapters for pipeline ports."""

from medical_kg_nlp.adapters.huggingface import (
    HuggingFaceCrossEncoderAdapter,
    HuggingFaceModelConfig,
    HuggingFaceSourceTokenClassifierAdapter,
    HuggingFaceTextEncoderAdapter,
    HuggingFaceTokenClassifierAdapter,
    OptionalModelDependencyError,
)
from medical_kg_nlp.adapters.hybrid import (
    HybridArbitrationPolicy,
    HybridEntityExtractorAdapter,
)
from medical_kg_nlp.adapters.medication import MedicationMentionEntityExtractorAdapter
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
    "HuggingFaceSourceTokenClassifierAdapter",
    "HuggingFaceTextEncoderAdapter",
    "HuggingFaceTokenClassifierAdapter",
    "HybridArbitrationPolicy",
    "HybridEntityExtractorAdapter",
    "KGValidatorAdapter",
    "MedicationMentionEntityExtractorAdapter",
    "OptionalModelDependencyError",
    "RuleAssertionClassifierAdapter",
    "RuleEntityExtractorAdapter",
    "RuleRelationExtractorAdapter",
]
