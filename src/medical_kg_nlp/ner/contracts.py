"""Protocols and immutable context for rule-based proposal extractors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from medical_kg_nlp.ner.medication_list_parser import MedicationListItem
from medical_kg_nlp.ner.proposal import EntityProposal

__all__ = [
    "ProposalExtractorPort",
    "RuleNerContext",
]


@dataclass(frozen=True, slots=True)
class RuleNerContext:
    """Read-only evidence available to one proposal extractor."""

    medication_items: tuple[MedicationListItem, ...] = ()
    foundation_proposals: tuple[EntityProposal, ...] = ()


class ProposalExtractorPort(Protocol):
    """Generate evidence without resolving or suppressing competing spans."""

    def propose(
        self,
        source_text: str,
        context: RuleNerContext,
    ) -> tuple[EntityProposal, ...]:
        """Return source-local proposals in raw-text coordinates."""
