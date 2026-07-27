from __future__ import annotations

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_rule_ner_trace_records_cross_source_overlap_decisions() -> None:
    text = "Tăng huyết áp đang điều trị."
    ner = RuleBasedNER(
        DictionaryStore.from_jsonl(
            "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl"
        )
    )

    result = ner.extract_with_trace(text)

    assert [(entity.text, entity.type) for entity in result.entities] == [
        ("Tăng huyết áp", EntityType.DISEASE)
    ]
    assert any(
        proposal.source == "dictionary_exact"
        and proposal.span == (0, len("Tăng huyết áp"))
        for proposal in result.trace.proposals
    )
    assert all(
        text[proposal.span[0] : proposal.span[1]]
        for proposal in result.trace.proposals
    )
    assert any(
        decision.accepted
        and decision.selected_type == EntityType.DISEASE
        and decision.reason == "selected_global_utility"
        for decision in result.trace.decisions
    )


def test_contextual_type_resolver_uses_explicit_symptom_and_diagnosis_cues() -> None:
    store = _dual_type_store("chóng mặt")
    ner = RuleBasedNER(store, disease_symptom_fallback="abstain")
    text = "Triệu chứng: chóng mặt. Chẩn đoán: chóng mặt. Không rõ chóng mặt."

    result = ner.extract_with_trace(text)

    assert [(entity.text, entity.type) for entity in result.entities] == [
        ("chóng mặt", EntityType.SYMPTOM),
        ("chóng mặt", EntityType.DISEASE),
    ]
    unresolved = result.unresolved_proposals
    assert len(unresolved) == 1
    assert text[unresolved[0].span[0] : unresolved[0].span[1]] == "chóng mặt"
    assert unresolved[0].feature("type_resolution") == "disease_symptom_context_missing"


def test_contextual_type_resolver_uses_sections_for_ambiguous_type_evidence() -> None:
    ner = RuleBasedNER(_dual_type_store("táo bón"))
    text = (
        "Bệnh lý mãn tính\n- táo bón\n"
        "Triệu chứng hiện tại\n- táo bón\n"
        "Các sự kiện trước khi nhập viện\n- táo bón\n"
        "Kết quả xét nghiệm\n- táo bón"
    )

    result = ner.extract_with_trace(text)

    assert [(entity.text, entity.type) for entity in result.entities] == [
        ("táo bón", EntityType.DISEASE),
        ("táo bón", EntityType.SYMPTOM),
        ("táo bón", EntityType.DISEASE),
        ("táo bón", EntityType.DISEASE),
    ]
    assert any(
        proposal.feature("type_resolution") == "explicit_symptom_section"
        for proposal in result.trace.proposals
    )


def test_contextual_type_resolver_does_not_invent_unproposed_type() -> None:
    store = DictionaryStore(
        [
            ConceptEntry(
                concept_id="D:1",
                code="K59.0",
                code_system=CodeSystem.ICD10,
                canonical_name="táo bón",
                semantic_type=EntityType.DISEASE,
            )
        ]
    )
    ner = RuleBasedNER(store)
    text = (
        "Bệnh lý mãn tính\n- táo bón\n"
        "Triệu chứng hiện tại\n- táo bón\n"
        "Các sự kiện trước khi nhập viện\n- táo bón\n"
        "Kết quả xét nghiệm\n- táo bón"
    )

    result = ner.extract_with_trace(text)

    assert [(entity.text, entity.type) for entity in result.entities] == [
        ("táo bón", EntityType.DISEASE),
        ("táo bón", EntityType.DISEASE),
        ("táo bón", EntityType.DISEASE),
        ("táo bón", EntityType.DISEASE),
    ]
    assert all(
        proposal.feature("type_resolution") == "unique_dictionary_type"
        for proposal in result.trace.proposals
        if proposal.source == "dictionary_exact"
    )


def test_rule_ner_keeps_medication_and_lab_sources_independent_until_resolution() -> None:
    text = "Dùng metoprolol 25mg. Creatinine 1.4 mg/dL."
    ner = RuleBasedNER(
        DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    )

    result = ner.extract_with_trace(text)
    proposal_sources = {
        (text[proposal.span[0] : proposal.span[1]], proposal.source)
        for proposal in result.trace.proposals
    }

    assert ("metoprolol", "dictionary_exact") in proposal_sources
    assert ("25mg", "medication_attribute") in proposal_sources
    assert ("Creatinine", "dictionary_exact") in proposal_sources
    assert ("1.4 mg/dL", "lab_anchor") in proposal_sources
    assert ("1.4 mg/dL", "regex_lab_result") in proposal_sources
    assert all(
        text[entity.span[0] : entity.span[1]] == entity.text
        for entity in result.entities
    )


def _dual_type_store(mention: str) -> DictionaryStore:
    return DictionaryStore(
        [
            ConceptEntry(
                concept_id="D:1",
                code="R42",
                code_system=CodeSystem.ICD10,
                canonical_name=mention,
                semantic_type=EntityType.DISEASE,
            ),
            ConceptEntry(
                concept_id="S:1",
                code="SYMPTOM",
                code_system=CodeSystem.LOCAL,
                canonical_name=mention,
                semantic_type=EntityType.SYMPTOM,
            ),
        ]
    )
