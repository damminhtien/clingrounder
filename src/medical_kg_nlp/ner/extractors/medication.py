"""Medication structure proposal sources."""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.ner.contracts import RuleNerContext
from medical_kg_nlp.ner.medication_attribute_extractor import MedicationAttributeExtractor
from medical_kg_nlp.ner.proposal import EntityProposal
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["MedicationAttributeProposalExtractor"]


@dataclass(frozen=True, slots=True)
class MedicationAttributeProposalExtractor:
    """Propose structured SIG components around every unresolved drug anchor."""

    implementation: MedicationAttributeExtractor

    def propose(
        self,
        source_text: str,
        context: RuleNerContext,
    ) -> tuple[EntityProposal, ...]:
        drug_anchors = [
            _anchor_entity(source_text, proposal)
            for proposal in context.foundation_proposals
            if proposal.entity_type == EntityType.DRUG
        ]
        attributes = self.implementation.extract(source_text, drug_anchors)
        return tuple(
            EntityProposal(
                span=entity.span,
                candidate_types=(entity.type,),
                source="medication_attribute",
                score=entity.confidence,
                evidence_ids=(f"medication_attribute:{entity.type.value}",),
            )
            for entity in attributes
        )


def _anchor_entity(source_text: str, proposal: EntityProposal) -> EntityAnnotation:
    start, end = proposal.span
    mention = source_text[start:end]
    return EntityAnnotation(
        id=f"proposal:{start}:{end}:DRUG",
        span=proposal.span,
        text=mention,
        normalized_text=normalize_for_match(mention),
        type=EntityType.DRUG,
        assertion=AssertionStatus.UNKNOWN,
        code_system=CodeSystem.NONE,
        confidence=proposal.score,
    )
