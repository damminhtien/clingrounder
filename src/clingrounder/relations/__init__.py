"""Typed relation extraction implementations and evidence-backed resources."""

from __future__ import annotations

from clingrounder.relations.knowledge import KnownRelation, KnownRelationRepository
from clingrounder.relations.rule_relations import RuleRelationExtractor

__all__ = ["KnownRelation", "KnownRelationRepository", "RuleRelationExtractor"]
