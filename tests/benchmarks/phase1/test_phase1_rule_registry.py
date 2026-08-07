from __future__ import annotations

import pytest

from clingrounder.benchmarks.phase1.phase1_rule_registry import (
    load_phase1_rule_registry,
    phase1_rule_registry_from_data,
)


def test_rule_registry_rejects_document_specific_selectors_recursively() -> None:
    payload = _registry(
        {
            "rule_id": "exclude.procedure",
            "stage": "strict_exclusion",
            "entity_type": "CHẨN_ĐOÁN",
            "normalized_mention": "phẫu thuật",
            "action": "block",
            "review_status": "reviewed",
            "provenance": {"document_ids": ["1"]},
        }
    )

    with pytest.raises(ValueError, match="forbidden document-specific field"):
        phase1_rule_registry_from_data(payload)


def test_rule_registry_rejects_absolute_span_and_multi_candidate_rules() -> None:
    span_rule = _registry(
        {
            "rule_id": "bad.span.rule",
            "stage": "strict_exclusion",
            "action": "block",
            "position": [1, 4],
        }
    )
    with pytest.raises(ValueError, match="forbidden document-specific field"):
        phase1_rule_registry_from_data(span_rule)

    candidate_rule = _registry(
        {
            "rule_id": "candidate.icd.ambiguous",
            "stage": "candidate_icd",
            "entity_type": "CHẨN_ĐOÁN",
            "normalized_mention": "bệnh",
            "action": "emit",
            "candidates": ["A00", "A01"],
        }
    )
    with pytest.raises(ValueError, match="at most one candidate"):
        phase1_rule_registry_from_data(candidate_rule)


def test_rule_registry_only_activates_reviewed_rules() -> None:
    registry = phase1_rule_registry_from_data(
        {
            "schema_version": "phase1-rule-registry.v1",
            "rules": [
                {
                    "rule_id": "assert.history.reviewed",
                    "stage": "assertion_history",
                    "entity_type": "CHẨN_ĐOÁN",
                    "action": "emit",
                    "left_regex": "tiền sử",
                    "assertions": ["isHistorical"],
                    "review_status": "reviewed",
                },
                {
                    "rule_id": "assert.history.draft",
                    "stage": "assertion_history",
                    "entity_type": "CHẨN_ĐOÁN",
                    "action": "emit",
                    "left_regex": "trước đây",
                    "assertions": ["isHistorical"],
                    "review_status": "draft",
                },
            ],
        }
    )

    assert [rule.rule_id for rule in registry.active_rules("assertion_history")] == [
        "assert.history.reviewed"
    ]


@pytest.mark.private
def test_local_top10_registry_keeps_discovered_rules_draft() -> None:
    registry = load_phase1_rule_registry("data/manual_gold/derived/top10-rule-registry.yaml")

    assert len(registry.rules) == 2
    assert registry.active_rules() == ()


def _registry(rule: dict[str, object]) -> dict[str, object]:
    return {"schema_version": "phase1-rule-registry.v1", "rules": [rule]}
