"""Anchored and regular-expression laboratory proposal sources."""

from __future__ import annotations

import re
from dataclasses import dataclass

from clingrounder.ner.contracts import RuleNerContext
from clingrounder.ner.lab_observation_extractor import LabObservationExtractor
from clingrounder.ner.proposal import EntityProposal
from clingrounder.schema.annotation import EntityAnnotation
from clingrounder.schema.types import AssertionStatus, CodeSystem, EntityType
from clingrounder.utils.text import normalize_for_match

__all__ = [
    "AnchoredLabProposalExtractor",
    "RegexLabProposalExtractor",
]


_LAB_VALUE_RE = re.compile(
    r"(?<!\w)\d+(?:[\.,]\d+)?\s?(?:mmol/L|mg/dL|g/dL|ng/mL|mEq/L|IU/L|U/L|%)(?!\w)",
    flags=re.IGNORECASE,
)
_BP_RE = re.compile(
    r"(?<!\w)BP\s*(?P<value>\d{2,3}/\d{2,3})(?!\w)",
    flags=re.IGNORECASE,
)
_VITAL_VALUE_RE = re.compile(
    r"(?<!\w)"
    r"(?:"
    r"huyết\s+áp(?:\s+tâm\s+(?:thu|trương))?|"
    r"nhịp\s+thở|"
    r"nhịp\s+tim|"
    r"nhiệt\s+độ|"
    r"thân\s+nhiệt|"
    r"spo2|"
    r"độ\s+bão\s+h[oò]a\s+oxy|"
    r"bão\s+h[oò]a\s+oxy"
    r")"
    r"(?:\s+là)?\s*"
    r"(?P<value>\d{2,5}(?:/\d{2,3})?(?:[\.,]\d+)?(?:-\d{2,3})?"
    r"(?:\s*%|\s*mmhg|\s*°?\s*c)?)"
    r"(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_HBA1C_RE = re.compile(
    r"(?<!\w)HbA1c\s*\d+(?:\.\d+)?%(?!\w)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AnchoredLabProposalExtractor:
    """Find result spans around every proposed lab-test anchor."""

    implementation: LabObservationExtractor

    def propose(
        self,
        source_text: str,
        context: RuleNerContext,
    ) -> tuple[EntityProposal, ...]:
        anchors = [
            _anchor_entity(source_text, proposal)
            for proposal in context.foundation_proposals
            if proposal.entity_type == EntityType.LAB_TEST
        ]
        disease_spans = tuple(
            proposal.span
            for proposal in context.foundation_proposals
            if proposal.entity_type == EntityType.DISEASE
        )
        results = tuple(
            entity
            for entity in self.implementation.extract(source_text, anchors)
            if not any(_contains(span, entity.span) for span in disease_spans)
        )
        return tuple(
            EntityProposal(
                span=entity.span,
                candidate_types=(EntityType.LAB_RESULT,),
                source="lab_anchor",
                score=entity.confidence,
                evidence_ids=("lab_anchor:result",),
                features=(("default_assertion", AssertionStatus.PRESENT.value),),
            )
            for entity in results
        )


@dataclass(frozen=True, slots=True)
class RegexLabProposalExtractor:
    """Propose self-describing vital and unit-bearing result spans."""

    def propose(
        self,
        source_text: str,
        context: RuleNerContext,
    ) -> tuple[EntityProposal, ...]:
        del context
        proposals: set[EntityProposal] = set()
        for rule_id, pattern in (
            ("hba1c", _HBA1C_RE),
            ("blood_pressure", _BP_RE),
            ("vital_value", _VITAL_VALUE_RE),
            ("unit_value", _LAB_VALUE_RE),
        ):
            for match in pattern.finditer(source_text):
                span = _result_span(match)
                proposals.add(
                    EntityProposal(
                        span=span,
                        candidate_types=(EntityType.LAB_RESULT,),
                        source="regex_lab_result",
                        score=0.8,
                        evidence_ids=(f"regex_lab_result:{rule_id}",),
                        features=(
                            ("default_assertion", AssertionStatus.PRESENT.value),
                        ),
                    )
                )
        return tuple(sorted(proposals, key=lambda item: item.span))


def _anchor_entity(source_text: str, proposal: EntityProposal) -> EntityAnnotation:
    start, end = proposal.span
    mention = source_text[start:end]
    return EntityAnnotation(
        id=f"proposal:{start}:{end}:LAB_TEST",
        span=proposal.span,
        text=mention,
        normalized_text=normalize_for_match(mention),
        type=EntityType.LAB_TEST,
        assertion=AssertionStatus.UNKNOWN,
        code_system=CodeSystem.NONE,
        confidence=proposal.score,
    )


def _result_span(match: re.Match[str]) -> tuple[int, int]:
    if "value" in match.re.groupindex:
        return match.span("value")
    return match.span()


def _contains(container: tuple[int, int], inner: tuple[int, int]) -> bool:
    """Return whether lexical disease evidence fully explains a would-be lab result."""

    return container[0] <= inner[0] and inner[1] <= container[1]
