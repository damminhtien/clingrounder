"""Backward-compatible entry point for configurable clinical section detection."""

from __future__ import annotations

from medical_kg_nlp.preprocessing.section_rules import (
    DEFAULT_SECTION_RULE_REGISTRY,
    RuleBasedSectionDetector,
    SectionRuleRegistry,
)
from medical_kg_nlp.schema.document import Section

__all__ = ["split_sections"]


def split_sections(
    text: str,
    registry: SectionRuleRegistry = DEFAULT_SECTION_RULE_REGISTRY,
) -> list[Section]:
    """Split ``text`` with the default or an injected section registry."""

    return RuleBasedSectionDetector(registry).detect(text)
