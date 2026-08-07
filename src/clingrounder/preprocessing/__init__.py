"""Offset-preserving normalization, sectioning, and sentence splitting."""

from __future__ import annotations

from clingrounder.preprocessing.offset_mapping import OffsetMappedText
from clingrounder.preprocessing.normalizer import (
    DEFAULT_NORMALIZATION_CONTRACT,
    NORMALIZATION_CONTRACT_VERSION,
    NormalizationContract,
)
from clingrounder.preprocessing.sentence_splitter import split_sentences
from clingrounder.preprocessing.section_rules import (
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
