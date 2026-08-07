"""Independent evidence sources used by the rule NER engine."""

from __future__ import annotations

from clingrounder.ner.extractors.boundary import ClinicalBoundaryProposalExtractor
from clingrounder.ner.extractors.contextual_alias import (
    ContextualAliasProposalExtractor,
)
from clingrounder.ner.extractors.dictionary import (
    ConcatenatedDrugProposalExtractor,
    DictionaryProposalExtractor,
)
from clingrounder.ner.extractors.laboratory import (
    AnchoredLabProposalExtractor,
    RegexLabProposalExtractor,
)
from clingrounder.ner.extractors.medication import MedicationAttributeProposalExtractor
from clingrounder.ner.extractors.structured_lab import (
    StructuredLabProposalExtractor,
)

__all__ = [
    "AnchoredLabProposalExtractor",
    "ClinicalBoundaryProposalExtractor",
    "ConcatenatedDrugProposalExtractor",
    "ContextualAliasProposalExtractor",
    "DictionaryProposalExtractor",
    "MedicationAttributeProposalExtractor",
    "RegexLabProposalExtractor",
    "StructuredLabProposalExtractor",
]
