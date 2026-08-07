from __future__ import annotations

from clingrounder.context.cue_loader import (
    AssertionRuleRegistry,
    cues_by_assertion,
    load_default_assertion_cues,
    section_priors_from_cues,
)
from clingrounder.schema.types import AssertionStatus


_SOURCE_CUES = load_default_assertion_cues()
ASSERTION_RULE_REGISTRY = AssertionRuleRegistry(_SOURCE_CUES)
_LEFT_CUES_BY_ASSERTION = cues_by_assertion(_SOURCE_CUES, scope="left")
_RIGHT_CUES_BY_ASSERTION = cues_by_assertion(_SOURCE_CUES, scope="right")
_BIDIRECTIONAL_CUES_BY_ASSERTION = cues_by_assertion(_SOURCE_CUES, scope="bidirectional")


def _directional_cues(
    assertion: AssertionStatus,
    *,
    direction: str,
) -> tuple[str, ...]:
    scoped = _LEFT_CUES_BY_ASSERTION if direction == "left" else _RIGHT_CUES_BY_ASSERTION
    values = (*scoped.get(assertion, ()), *_BIDIRECTIONAL_CUES_BY_ASSERTION.get(assertion, ()))
    return tuple(dict.fromkeys(values))


POSSIBLE_LEFT_CUES = _directional_cues(AssertionStatus.POSSIBLE, direction="left")
POSSIBLE_RIGHT_CUES = _directional_cues(AssertionStatus.POSSIBLE, direction="right")
NEGATION_LEFT_CUES = _directional_cues(AssertionStatus.NEGATED, direction="left")
NEGATION_RIGHT_CUES = _directional_cues(AssertionStatus.NEGATED, direction="right")
HISTORICAL_LEFT_CUES = _directional_cues(AssertionStatus.HISTORICAL, direction="left")
HISTORICAL_RIGHT_CUES = _directional_cues(AssertionStatus.HISTORICAL, direction="right")
FAMILY_LEFT_CUES = _directional_cues(AssertionStatus.FAMILY, direction="left")
FAMILY_RIGHT_CUES = _directional_cues(AssertionStatus.FAMILY, direction="right")
PLANNED_LEFT_CUES = _directional_cues(AssertionStatus.PLANNED, direction="left")
PLANNED_RIGHT_CUES = _directional_cues(AssertionStatus.PLANNED, direction="right")
RESOLVED_LEFT_CUES = _directional_cues(AssertionStatus.RESOLVED, direction="left")
RESOLVED_RIGHT_CUES = _directional_cues(AssertionStatus.RESOLVED, direction="right")

POSSIBLE_CUES = tuple(dict.fromkeys((*POSSIBLE_LEFT_CUES, *POSSIBLE_RIGHT_CUES)))
NEGATION_CUES = tuple(dict.fromkeys((*NEGATION_LEFT_CUES, *NEGATION_RIGHT_CUES)))
HISTORICAL_CUES = tuple(dict.fromkeys((*HISTORICAL_LEFT_CUES, *HISTORICAL_RIGHT_CUES)))
FAMILY_CUES = tuple(dict.fromkeys((*FAMILY_LEFT_CUES, *FAMILY_RIGHT_CUES)))
PLANNED_CUES = tuple(dict.fromkeys((*PLANNED_LEFT_CUES, *PLANNED_RIGHT_CUES)))
RESOLVED_CUES = tuple(dict.fromkeys((*RESOLVED_LEFT_CUES, *RESOLVED_RIGHT_CUES)))

SECTION_PRIORS = section_priors_from_cues(_SOURCE_CUES)
