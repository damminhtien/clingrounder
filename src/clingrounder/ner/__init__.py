"""Entity extraction implementations and span-oriented helpers."""

from __future__ import annotations

from clingrounder.ner.contracts import ProposalExtractorPort, RuleNerContext
from clingrounder.ner.document_structure import (
    DocumentGenre,
    DocumentStructure,
    DocumentStructureAnalyzer,
    SectionKind,
)
from clingrounder.ner.proposal import EntityProposal, ProposalDecision, RuleNerTrace
from clingrounder.ner.rule_engine import RuleNerEngine, RuleNerEngineResult
from clingrounder.ner.rule_ner import RuleBasedNER
from clingrounder.ner.span_resolver import EvidenceWeightedSpanResolver

__all__ = [
    "EntityProposal",
    "EvidenceWeightedSpanResolver",
    "DocumentGenre",
    "DocumentStructure",
    "DocumentStructureAnalyzer",
    "ProposalDecision",
    "ProposalExtractorPort",
    "RuleBasedNER",
    "RuleNerContext",
    "RuleNerEngine",
    "RuleNerEngineResult",
    "RuleNerTrace",
    "SectionKind",
]
