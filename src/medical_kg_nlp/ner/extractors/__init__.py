"""Independent evidence sources used by the rule NER engine."""

from __future__ import annotations

from medical_kg_nlp.ner.extractors.dictionary import (
    ConcatenatedDrugProposalExtractor,
    DictionaryProposalExtractor,
)
from medical_kg_nlp.ner.extractors.laboratory import (
    AnchoredLabProposalExtractor,
    RegexLabProposalExtractor,
)
from medical_kg_nlp.ner.extractors.medication import MedicationAttributeProposalExtractor

__all__ = [
    "AnchoredLabProposalExtractor",
    "ConcatenatedDrugProposalExtractor",
    "DictionaryProposalExtractor",
    "MedicationAttributeProposalExtractor",
    "RegexLabProposalExtractor",
]
