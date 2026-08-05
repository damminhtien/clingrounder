"""Typed relation extraction implementations and evidence-backed resources."""

from __future__ import annotations

from medical_kg_nlp.relations.knowledge import KnownRelation, KnownRelationRepository
from medical_kg_nlp.relations.rule_relations import RuleRelationExtractor

__all__ = ["KnownRelation", "KnownRelationRepository", "RuleRelationExtractor"]
