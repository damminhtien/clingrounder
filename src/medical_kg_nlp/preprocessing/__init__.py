"""Offset-preserving normalization, sectioning, and sentence splitting."""

from __future__ import annotations

from medical_kg_nlp.preprocessing.offset_mapping import OffsetMappedText
from medical_kg_nlp.preprocessing.normalizer import (
    DEFAULT_NORMALIZATION_CONTRACT,
    NORMALIZATION_CONTRACT_VERSION,
    NormalizationContract,
)
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.preprocessing.section_rules import (
    DEFAULT_SECTION_RULE_REGISTRY,
    RuleBasedSectionDetector,
    SectionRule,
    SectionRuleRegistry,
    split_sections,
)

__all__ = [
    "DEFAULT_NORMALIZATION_CONTRACT",
    "NORMALIZATION_CONTRACT_VERSION",
    "NormalizationContract",
    "OffsetMappedText",
    "split_sections",
    "DEFAULT_SECTION_RULE_REGISTRY",
    "RuleBasedSectionDetector",
    "SectionRule",
    "SectionRuleRegistry",
    "split_sentences",
]
