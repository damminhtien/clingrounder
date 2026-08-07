"""Built-in adapters for pipeline ports."""

from clingrounder.adapters.generative import GenerativeListwiseRerankerAdapter
from clingrounder.adapters.huggingface import (
    HuggingFaceCrossEncoderAdapter,
    HuggingFaceModelConfig,
    HuggingFaceMulticlassTextClassifierAdapter,
    HuggingFaceSourceTokenClassifierAdapter,
    HuggingFaceTextEncoderAdapter,
    HuggingFaceTokenClassifierAdapter,
    OptionalModelDependencyError,
)
from clingrounder.adapters.hybrid import (
    HybridArbitrationPolicy,
    HybridEntityExtractorAdapter,
)
from clingrounder.adapters.medication import MedicationMentionEntityExtractorAdapter
from clingrounder.adapters.rules import (
    DictionaryCandidateAdapter,
    KGValidatorAdapter,
    RuleAssertionClassifierAdapter,
    RuleEntityExtractorAdapter,
    RuleRelationExtractorAdapter,
)

__all__ = [
    "DictionaryCandidateAdapter",
    "GenerativeListwiseRerankerAdapter",
    "HuggingFaceCrossEncoderAdapter",
    "HuggingFaceModelConfig",
    "HuggingFaceMulticlassTextClassifierAdapter",
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
