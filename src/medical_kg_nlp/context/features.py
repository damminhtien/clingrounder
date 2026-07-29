"""Task-neutral sparse features for assertion model adapters.

Feature extraction is intentionally separate from assertion decoding, following
the cTAKES architecture. A rule decision can be a model feature without becoming
the model label or overriding another independent assertion attribute.
"""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.context.modifier_graph import ContextGraph
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["AssertionModelFeatureExtractor"]


@dataclass(frozen=True, slots=True)
class AssertionModelFeatureExtractor:
    """Convert entity, section, and modifier evidence to bounded sparse features."""

    max_rule_features: int = 8

    def __post_init__(self) -> None:
        if self.max_rule_features < 1:
            raise ValueError("max_rule_features must be positive.")

    def extract(
        self,
        entity: EntityAnnotation,
        sentence: Sentence,
        graph: ContextGraph,
    ) -> dict[str, float]:
        """Return deterministic numeric features for one target."""

        if entity.id not in {target.entity_id for target in graph.targets}:
            raise ValueError(f"Entity {entity.id!r} is absent from the context graph.")
        features = {
            "bias": 1.0,
            f"entity_type:{entity.type.value}": 1.0,
            f"target_position:{_position_bucket(entity, sentence)}": 1.0,
            f"target_length:{_length_bucket(len(entity.text))}": 1.0,
        }
        section = normalize_for_match(sentence.section_title or "")
        if section:
            features[f"section:{section}"] = 1.0

        edges = graph.edges_for(entity.id)
        features["modifier_count"] = float(len(edges))
        if not edges:
            features["no_modifier"] = 1.0
            return features

        modifiers = {node.node_id: node for node in graph.modifiers}
        minimum_distance = min(edge.distance for edge in edges)
        features["nearest_modifier_distance"] = float(minimum_distance)
        features[f"nearest_modifier_bucket:{_distance_bucket(minimum_distance)}"] = 1.0

        for edge in edges:
            features[f"assertion:{edge.assertion.value}"] = 1.0
            features[f"scope:{edge.scope}"] = 1.0
            features[f"assertion_scope:{edge.assertion.value}:{edge.scope}"] = 1.0
        # SCALING: high-cardinality rule IDs are bounded; semantic assertion and
        # direction features above remain available for every edge.
        ranked = sorted(
            edges,
            key=lambda edge: (edge.distance, edge.modifier_id),
        )[: self.max_rule_features]
        for edge in ranked:
            modifier = modifiers[edge.modifier_id]
            features[f"rule:{modifier.rule_id}"] = 1.0
        return features


def _position_bucket(entity: EntityAnnotation, sentence: Sentence) -> str:
    denominator = max(1, sentence.span[1] - sentence.span[0])
    midpoint = ((entity.span[0] + entity.span[1]) / 2) - sentence.span[0]
    ratio = midpoint / denominator
    if ratio < 1 / 3:
        return "start"
    if ratio < 2 / 3:
        return "middle"
    return "end"


def _length_bucket(length: int) -> str:
    if length <= 4:
        return "short"
    if length <= 20:
        return "medium"
    return "long"


def _distance_bucket(distance: int) -> str:
    if distance <= 4:
        return "near"
    if distance <= 20:
        return "local"
    return "remote"
