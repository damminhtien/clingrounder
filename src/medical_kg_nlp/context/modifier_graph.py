"""Modifier-target evidence graph for clinical assertions.

This module adopts the graph boundary from medspaCy without depending on spaCy.
The graph records why a modifier affected a target; assertion policy remains in
the classifier and all graph spans stay in raw source coordinates.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TypeAlias

from medical_kg_nlp.schema.annotation import (
    AssertionEvidence,
    AssertionFeatures,
    EntityAnnotation,
)
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.schema.types import AssertionStatus, EntityType

__all__ = [
    "AssertionDecision",
    "ContextEdge",
    "ContextGraph",
    "ContextModifierNode",
    "ContextTargetNode",
    "build_context_graph",
]

AssertionDecision: TypeAlias = tuple[
    AssertionFeatures,
    tuple[AssertionEvidence, ...],
]


@dataclass(frozen=True, slots=True)
class ContextModifierNode:
    """One matched cue or section prior used by at least one decision."""

    node_id: str
    rule_id: str
    assertion: AssertionStatus
    cue: str
    scope: str
    span: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class ContextTargetNode:
    """One entity considered by assertion classification."""

    entity_id: str
    span: tuple[int, int]
    entity_type: EntityType


@dataclass(frozen=True, slots=True)
class ContextEdge:
    """A directed modifier-to-target assertion decision."""

    modifier_id: str
    target_id: str
    assertion: AssertionStatus
    scope: str
    distance: int


@dataclass(frozen=True, slots=True)
class ContextGraph:
    """Immutable evidence graph for one sentence."""

    sentence_span: tuple[int, int]
    modifiers: tuple[ContextModifierNode, ...]
    targets: tuple[ContextTargetNode, ...]
    edges: tuple[ContextEdge, ...]

    def edges_for(self, entity_id: str) -> tuple[ContextEdge, ...]:
        return tuple(edge for edge in self.edges if edge.target_id == entity_id)


def build_context_graph(
    sentence: Sentence,
    entities: list[EntityAnnotation],
    decisions: dict[str, AssertionDecision],
) -> ContextGraph:
    """Project classifier evidence onto explicit raw-coordinate graph nodes."""

    target_ids = [entity.id for entity in entities]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Context graph target entity IDs must be unique.")
    if set(decisions) != set(target_ids):
        raise ValueError("Context graph decisions must cover every target exactly once.")

    targets = tuple(
        ContextTargetNode(
            entity_id=entity.id,
            span=entity.span,
            entity_type=entity.type,
        )
        for entity in entities
    )
    modifiers: dict[str, ContextModifierNode] = {}
    edges: list[ContextEdge] = []
    for entity in entities:
        _validate_target(sentence, entity)
        _, evidence_items = decisions[entity.id]
        for evidence in evidence_items:
            cue_span = _locate_evidence(sentence, entity, evidence)
            node_id = _modifier_id(evidence, cue_span)
            modifiers.setdefault(
                node_id,
                ContextModifierNode(
                    node_id=node_id,
                    rule_id=evidence.rule_id,
                    assertion=evidence.assertion,
                    cue=evidence.cue,
                    scope=evidence.scope,
                    span=cue_span,
                ),
            )
            edges.append(
                ContextEdge(
                    modifier_id=node_id,
                    target_id=entity.id,
                    assertion=evidence.assertion,
                    scope=evidence.scope,
                    distance=_distance(cue_span, entity.span),
                )
            )

    ordered_modifiers = tuple(
        sorted(
            modifiers.values(),
            key=lambda node: (
                node.span is None,
                node.span or sentence.span,
                node.rule_id,
                node.scope,
            ),
        )
    )
    ordered_edges = tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.target_id,
                edge.distance,
                edge.assertion.value,
                edge.modifier_id,
            ),
        )
    )
    return ContextGraph(
        sentence_span=sentence.span,
        modifiers=ordered_modifiers,
        targets=targets,
        edges=ordered_edges,
    )


def _validate_target(sentence: Sentence, entity: EntityAnnotation) -> None:
    if not (
        sentence.span[0] <= entity.span[0] < entity.span[1] <= sentence.span[1]
    ):
        raise ValueError(
            f"Entity {entity.id!r} span {entity.span} is outside sentence {sentence.span}."
        )
    local_start = entity.span[0] - sentence.span[0]
    local_end = entity.span[1] - sentence.span[0]
    if sentence.text[local_start:local_end] != entity.text:
        raise ValueError(f"Entity {entity.id!r} does not match sentence source text.")


def _locate_evidence(
    sentence: Sentence,
    entity: EntityAnnotation,
    evidence: AssertionEvidence,
) -> tuple[int, int] | None:
    if evidence.scope == "section_prior":
        return None
    pattern = re.compile(
        rf"(?<!\w){re.escape(evidence.cue)}(?!\w)",
        flags=re.IGNORECASE | re.UNICODE,
    )
    entity_start = entity.span[0] - sentence.span[0]
    entity_end = entity.span[1] - sentence.span[0]
    matches = list(pattern.finditer(sentence.text))
    if evidence.scope == "left":
        eligible = [match for match in matches if match.end() <= entity_start]
        selected = max(eligible, key=lambda match: match.end(), default=None)
    elif evidence.scope == "right":
        eligible = [match for match in matches if match.start() >= entity_end]
        selected = min(eligible, key=lambda match: match.start(), default=None)
    else:
        raise ValueError(f"Unsupported assertion evidence scope: {evidence.scope!r}.")
    if selected is None:
        raise ValueError(
            f"Cannot project assertion cue {evidence.cue!r} for entity {entity.id!r}."
        )
    span = (
        sentence.span[0] + selected.start(),
        sentence.span[0] + selected.end(),
    )
    # INVARIANT: a context node references the exact cue occurrence in source text.
    if sentence.text[selected.start() : selected.end()].casefold() != evidence.cue.casefold():
        raise AssertionError("Projected assertion cue does not match source text.")
    return span


def _modifier_id(
    evidence: AssertionEvidence,
    span: tuple[int, int] | None,
) -> str:
    position = "section" if span is None else f"{span[0]}:{span[1]}"
    payload = (
        f"{evidence.rule_id}\0{evidence.assertion.value}\0"
        f"{evidence.scope}\0{position}"
    )
    return f"M-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _distance(
    modifier_span: tuple[int, int] | None,
    target_span: tuple[int, int],
) -> int:
    if modifier_span is None:
        return 0
    if modifier_span[1] <= target_span[0]:
        return target_span[0] - modifier_span[1]
    if target_span[1] <= modifier_span[0]:
        return modifier_span[0] - target_span[1]
    return 0
