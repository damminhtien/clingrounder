"""Proposal-first orchestration for deterministic rule-based NER."""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.ner.contracts import ProposalExtractorPort, RuleNerContext
from medical_kg_nlp.ner.document_structure import DocumentStructureAnalyzer
from medical_kg_nlp.ner.medication_list_parser import MedicationListParser
from medical_kg_nlp.ner.medication_mention_parser import MedicationMentionParser
from medical_kg_nlp.ner.proposal import EntityProposal, RuleNerTrace
from medical_kg_nlp.ner.span_resolver import EvidenceWeightedSpanResolver
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "RuleNerEngine",
    "RuleNerEngineResult",
]


@dataclass(frozen=True, slots=True)
class RuleNerEngineResult:
    """Resolved entities, unresolved evidence, and full decision lineage."""

    entities: tuple[EntityAnnotation, ...]
    unresolved_proposals: tuple[EntityProposal, ...]
    trace: RuleNerTrace


@dataclass(frozen=True, slots=True)
class RuleNerEngine:
    """Compose independent rule sources and resolve all span conflicts once."""

    foundation_extractors: tuple[ProposalExtractorPort, ...]
    dependent_extractors: tuple[ProposalExtractorPort, ...]
    span_resolver: EvidenceWeightedSpanResolver
    medication_mentions: MedicationMentionParser
    medication_lists: MedicationListParser
    document_structure: DocumentStructureAnalyzer

    def extract(self, source_text: str) -> RuleNerEngineResult:
        structure = self.document_structure.analyze(source_text)
        medication_items = self.medication_lists.items(source_text)
        initial_context = RuleNerContext(
            medication_items=medication_items,
            structure=structure,
        )
        foundation = tuple(
            proposal
            for extractor in self.foundation_extractors
            for proposal in extractor.propose(source_text, initial_context)
        )
        dependent_context = RuleNerContext(
            medication_items=medication_items,
            foundation_proposals=foundation,
            structure=structure,
        )
        dependent = tuple(
            proposal
            for extractor in self.dependent_extractors
            for proposal in extractor.propose(source_text, dependent_context)
        )
        proposals = tuple(sorted((*foundation, *dependent), key=_proposal_order))

        # INVARIANT: no extractor receives accepted/rejected spans. All cross-source conflicts are
        # decided here, after the complete evidence set has been materialized.
        resolution = self.span_resolver.resolve(proposals)
        entities = [
            _entity_from_proposal(source_text, proposal)
            for proposal in resolution.selected
        ]
        for entity in entities:
            if entity.type == EntityType.DRUG:
                entity.medication_mention = self.medication_mentions.parse(
                    source_text,
                    entity.span,
                )
        entities = self.medication_lists.adjudicate(source_text, entities)
        entities.sort(key=lambda entity: (entity.span[0], entity.span[1], entity.type.value))
        for index, entity in enumerate(entities, start=1):
            entity.id = f"E{index}"
            entity.validate_offsets(source_text)

        unresolved = tuple(
            proposal for proposal in proposals if proposal.entity_type is None
        )
        return RuleNerEngineResult(
            entities=tuple(entities),
            unresolved_proposals=unresolved,
            trace=RuleNerTrace(
                proposals=proposals,
                decisions=resolution.decisions,
            ),
        )


def _entity_from_proposal(
    source_text: str,
    proposal: EntityProposal,
) -> EntityAnnotation:
    entity_type = proposal.entity_type
    if entity_type is None:
        raise ValueError("Cannot build an entity from an unresolved proposal")
    start, end = proposal.span
    mention = source_text[start:end]
    default_assertion = proposal.feature("default_assertion")
    assertion = (
        AssertionStatus(default_assertion)
        if default_assertion is not None
        else AssertionStatus.UNKNOWN
    )
    return EntityAnnotation(
        id="",
        span=proposal.span,
        text=mention,
        normalized_text=normalize_for_match(mention),
        type=entity_type,
        assertion=assertion,
        code_system=CodeSystem.NONE,
        code=None,
        confidence=proposal.score,
        candidates=[],
        medication_mention=proposal.medication_mention,
    )


def _proposal_order(
    proposal: EntityProposal,
) -> tuple[int, int, str, str]:
    return (
        proposal.span[0],
        proposal.span[1],
        ",".join(item.value for item in proposal.candidate_types),
        proposal.source,
    )
