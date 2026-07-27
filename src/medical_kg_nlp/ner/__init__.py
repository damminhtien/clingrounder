"""Entity extraction implementations and span-oriented helpers."""

from __future__ import annotations

from medical_kg_nlp.ner.contracts import ProposalExtractorPort, RuleNerContext
from medical_kg_nlp.ner.proposal import EntityProposal, ProposalDecision, RuleNerTrace
from medical_kg_nlp.ner.rule_engine import RuleNerEngine, RuleNerEngineResult
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.ner.span_resolver import EvidenceWeightedSpanResolver

__all__ = [
    "EntityProposal",
    "EvidenceWeightedSpanResolver",
    "ProposalDecision",
    "ProposalExtractorPort",
    "RuleBasedNER",
    "RuleNerContext",
    "RuleNerEngine",
    "RuleNerEngineResult",
    "RuleNerTrace",
]
