"""Compile train-supported Phase 1 aliases into generic context-gated rules."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from clingrounder.mining.lexicon import MentionInventoryEntry
from clingrounder.ner.extractors.contextual_alias import ContextGate
from clingrounder.benchmarks.phase1.ontology import PHASE1_RULE_BY_TYPE
from clingrounder.schema.types import EntityType
from clingrounder.utils.text import (
    normalize_for_match,
    strip_vietnamese_tones,
)

__all__ = [
    "Phase1ContextualAliasCompilation",
    "compile_phase1_contextual_alias_rules",
]


@dataclass(frozen=True, slots=True)
class Phase1ContextualAliasCompilation:
    """Runtime artifact plus auditable decisions for every reviewed alias."""

    artifact: dict[str, Any]
    decisions: tuple[dict[str, Any], ...]
    report: dict[str, Any]


_CONTEXTS_BY_TYPE: dict[EntityType, tuple[ContextGate, ...]] = {
    EntityType.SYMPTOM: tuple(
        sorted(
            {
                ContextGate.MEDICATION_INDICATION,
                ContextGate.STANDALONE_LIST_ITEM,
                ContextGate.SYMPTOM_PREDICATE,
                ContextGate.SYMPTOM_SECTION,
            }
        )
    ),
}
_NUMERIC_ONLY_RE = re.compile(r"^[\d\s.,<>%/+*-]+$")


def compile_phase1_contextual_alias_rules(
    annotation_policy: Mapping[str, Any],
    inventory: Sequence[MentionInventoryEntry],
    *,
    inventory_sha256: str,
) -> Phase1ContextualAliasCompilation:
    """Compile only aliases backed by the frozen train inventory.

    Numeric/result aliases remain owned by the lab grammar. This compiler currently promotes
    symptom aliases because they provide new recognition evidence; broad lab-test aliases already
    exist in the controlled dictionary and need a separate precision-block experiment.
    """

    if len(inventory_sha256) != 64:
        raise ValueError("Contextual alias compiler requires an inventory SHA-256")
    aliases = annotation_policy.get("aliases")
    if aliases is not None and not isinstance(aliases, Mapping):
        raise ValueError("Phase 1 annotation policy aliases must be an object")
    raw_context = aliases.get("context_required", {}) if isinstance(aliases, Mapping) else {}
    if not isinstance(raw_context, Mapping):
        raise ValueError("Phase 1 aliases.context_required must be an object")

    train_entries = {
        (entry.source_label, entry.normalized_mention): entry
        for entry in inventory
        if entry.source_label is not None
    }
    rules: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for source_label, raw_aliases in sorted(raw_context.items()):
        label = str(source_label)
        phase1_rule = PHASE1_RULE_BY_TYPE.get(label)
        if phase1_rule is None:
            raise ValueError(f"Unknown Phase 1 contextual-alias type: {label!r}")
        if not isinstance(raw_aliases, Sequence) or isinstance(raw_aliases, str):
            raise ValueError(f"Contextual aliases for {label!r} must be an array")
        entity_type = phase1_rule.internal_type
        for raw_alias in raw_aliases:
            alias = str(raw_alias).strip()
            normalized = normalize_for_match(alias)
            entry = train_entries.get((label, normalized))
            reason = _eligibility_reason(alias, entity_type, entry)
            decision = {
                "source_label": label,
                "normalized_alias": normalized,
                "entity_type": entity_type.value,
                "eligible": reason == "eligible",
                "reason": reason,
                "occurrence_count": entry.occurrence_count if entry is not None else 0,
                "document_count": entry.document_count if entry is not None else 0,
            }
            decisions.append(decision)
            if reason != "eligible":
                continue
            match_modes = ["exact"]
            if strip_vietnamese_tones(normalized) != normalized:
                match_modes.append("toneless")
            rules.append(
                {
                    "rule_id": _rule_id(label, normalized),
                    "alias": alias,
                    "entity_type": entity_type.value,
                    "required_any": [
                        gate.value for gate in _CONTEXTS_BY_TYPE[entity_type]
                    ],
                    "match_modes": sorted(match_modes),
                    # This is an emission prior, not a calibrated probability. Source/type
                    # calibration consumes train traces in a separate experiment.
                    "score": 0.72,
                    "provenance": "phase1_manual_gold_train_context_required",
                }
            )

    ordered_rules = sorted(rules, key=lambda rule: str(rule["rule_id"]))
    reason_counts: dict[str, int] = {}
    for decision in decisions:
        reason = str(decision["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    artifact = {
        "schema_version": "contextual-alias-rules.v1",
        "policy_id": "phase1-manual-gold-train-contextual-alias-v1",
        "inventory_sha256": inventory_sha256,
        "source_split": "train",
        "rules": ordered_rules,
    }
    return Phase1ContextualAliasCompilation(
        artifact=artifact,
        decisions=tuple(decisions),
        report={
            "reviewed_alias_count": len(decisions),
            "runtime_rule_count": len(ordered_rules),
            "reason_counts": dict(sorted(reason_counts.items())),
            "entity_type_counts": {
                entity_type.value: sum(
                    rule["entity_type"] == entity_type.value for rule in ordered_rules
                )
                for entity_type in sorted(_CONTEXTS_BY_TYPE, key=lambda item: item.value)
            },
        },
    )


def _eligibility_reason(
    alias: str,
    entity_type: EntityType,
    entry: MentionInventoryEntry | None,
) -> str:
    if not alias or not normalize_for_match(alias):
        return "empty_alias"
    if _NUMERIC_ONLY_RE.fullmatch(alias):
        return "numeric_owned_by_lab_grammar"
    if entry is None:
        return "absent_from_train_inventory"
    if entity_type not in _CONTEXTS_BY_TYPE:
        return "type_owned_by_existing_source"
    return "eligible"


def _rule_id(source_label: str, normalized_alias: str) -> str:
    identity = f"{source_label}\0{normalized_alias}".encode("utf-8")
    return f"phase1.context.{hashlib.sha256(identity).hexdigest()[:16]}"
