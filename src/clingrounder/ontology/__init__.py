"""Reusable ontology-adjacent rule contracts."""

from clingrounder.ontology.false_positive import (
    FalsePositiveRule,
    load_false_positive_rules,
)

__all__ = ["FalsePositiveRule", "load_false_positive_rules"]
