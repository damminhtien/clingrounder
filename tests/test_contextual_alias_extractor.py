from __future__ import annotations

import pytest

from medical_kg_nlp.ner.contracts import RuleNerContext
from medical_kg_nlp.ner.document_structure import DocumentStructureAnalyzer
from medical_kg_nlp.ner.extractors.contextual_alias import (
    ContextGate,
    ContextualAliasProposalExtractor,
    ContextualAliasRule,
    load_contextual_alias_rules,
)
from medical_kg_nlp.ner.medication_list_parser import MedicationListItem
from medical_kg_nlp.schema.types import EntityType


def test_contextual_alias_requires_reviewed_symptom_structure() -> None:
    text = "Triệu chứng hiện tại:\n- đau\nGiải thích đau là một từ phổ biến."
    extractor = ContextualAliasProposalExtractor((_pain_rule(),))
    context = RuleNerContext(
        structure=DocumentStructureAnalyzer().analyze(text),
    )

    proposals = extractor.propose(text, context)

    assert [(text[slice(*item.span)], item.entity_type) for item in proposals] == [
        ("đau", EntityType.SYMPTOM)
    ]
    assert proposals[0].feature("context_gates") == (
        "standalone_list_item,symptom_section"
    )


def test_contextual_alias_allows_short_toneless_only_inside_gate() -> None:
    text = "Triệu chứng hiện tại:\n- dau\nMã DAU không phải triệu chứng."
    extractor = ContextualAliasProposalExtractor((_pain_rule(),))
    context = RuleNerContext(
        structure=DocumentStructureAnalyzer().analyze(text),
    )

    proposals = extractor.propose(text, context)

    assert [text[slice(*item.span)] for item in proposals] == ["dau"]
    assert proposals[0].feature("match_kind") == "toneless"


def test_contextual_alias_uses_medication_indication_scope() -> None:
    text = "1. acetaminophen 500 mg po prn điều trị đau"
    indication_start = text.index("đau")
    context = RuleNerContext(
        medication_items=(
            MedicationListItem(
                medication_span=(3, text.index(" điều trị")),
                indication_span=(indication_start, len(text)),
            ),
        ),
        structure=DocumentStructureAnalyzer().analyze(text),
    )

    proposals = ContextualAliasProposalExtractor((_pain_rule(),)).propose(text, context)

    assert [text[slice(*item.span)] for item in proposals] == ["đau"]
    assert proposals[0].feature("context_gates") == "medication_indication"


def test_contextual_alias_loader_rejects_document_specific_fields(tmp_path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """\
schema_version: contextual-alias-rules.v1
rules:
  - rule_id: symptom.pain
    alias: đau
    entity_type: SYMPTOM
    required_any: [symptom_section]
    match_modes: [exact]
    document_id: "1"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="document-specific"):
        load_contextual_alias_rules(path)


def _pain_rule() -> ContextualAliasRule:
    return ContextualAliasRule(
        rule_id="symptom.pain",
        alias="đau",
        entity_type=EntityType.SYMPTOM,
        required_any=tuple(
            sorted(
                {
                    ContextGate.MEDICATION_INDICATION,
                    ContextGate.STANDALONE_LIST_ITEM,
                    ContextGate.SYMPTOM_PREDICATE,
                    ContextGate.SYMPTOM_SECTION,
                }
            )
        ),
        match_modes=("exact", "toneless"),
        score=0.72,
        provenance="reviewed_train",
    )
