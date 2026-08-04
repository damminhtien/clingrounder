"""Tests for genre-aware, cross-fitted mention-code memory."""

from __future__ import annotations

import math

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.linking.mention_code_memory import (
    MentionCodeMemoryObservation,
    MentionCodeMemoryRetrieverAdapter,
    build_cross_fitted_mention_code_memory,
    build_mention_code_memory,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.memory import InMemoryTerminologyRepository


def test_memory_profiles_support_probability_and_entropy() -> None:
    memory = build_mention_code_memory(
        (
            _observation("a", "I10"),
            _observation("b", "I10"),
            _observation("c", "I11"),
        )
    )

    record = memory.lookup("THA", EntityType.DISEASE, "clinical_note")

    assert record is not None
    assert record.document_support == 3
    assert record.most_common.code == "I10"
    assert record.most_common_probability == 2 / 3
    assert math.isclose(record.entropy, 0.9182958340544896)


def test_high_confidence_and_prior_tiers_are_disjoint() -> None:
    repository = _repository()
    memory = build_mention_code_memory(
        (_observation("a", "I10"), _observation("b", "I10"))
    )
    terminal = MentionCodeMemoryRetrieverAdapter(
        memory,
        repository,
        genre="clinical_note",
    )
    prior = MentionCodeMemoryRetrieverAdapter(
        memory,
        repository,
        genre="clinical_note",
        high_confidence_only=False,
    )

    assert terminal.terminal_on_match is True
    assert [item.code for item in terminal.retrieve("tha", EntityType.DISEASE, "", 5)] == [
        "I10"
    ]
    assert prior.retrieve("tha", EntityType.DISEASE, "", 5) == []


def test_ambiguous_memory_is_prior_only() -> None:
    memory = build_mention_code_memory(
        (_observation("a", "I10"), _observation("b", "I11"))
    )
    adapter = MentionCodeMemoryRetrieverAdapter(
        memory,
        _repository(),
        genre="clinical_note",
        high_confidence_only=False,
    )

    candidates = adapter.retrieve("tha", EntityType.DISEASE, "", 5)

    assert {item.code for item in candidates} == {"I10", "I11"}
    assert all(item.source == "mention_memory_prior" for item in candidates)


def test_cross_fit_excludes_every_observation_from_held_out_document() -> None:
    observations = (
        _observation("a", "I10"),
        _observation("a", "I10"),
        _observation("b", "I11"),
        _observation("c", "I10"),
    )
    cross_fitted = build_cross_fitted_mention_code_memory(
        observations,
        {"a": 0, "b": 1, "c": 1},
    )

    record_for_a = cross_fitted.memory_for_document("a").lookup(
        "tha", EntityType.DISEASE, "clinical_note"
    )

    assert record_for_a is not None
    assert dict((identity.code, count) for identity, count in record_for_a.code_counts) == {
        "I10": 1,
        "I11": 1,
    }
    assert record_for_a.document_support == 2


def _observation(document_id: str, code: str) -> MentionCodeMemoryObservation:
    return MentionCodeMemoryObservation(
        document_id=document_id,
        mention="THA",
        entity_type=EntityType.DISEASE,
        genre="clinical_note",
        code_system=CodeSystem.ICD10,
        code=code,
    )


def _repository() -> InMemoryTerminologyRepository:
    entries = [
        ConceptEntry(
            concept_id=f"ICD:{code}",
            code=code,
            code_system=CodeSystem.ICD10,
            canonical_name=code,
            semantic_type=EntityType.DISEASE,
        )
        for code in ("I10", "I11")
    ]
    return InMemoryTerminologyRepository(DictionaryStore(entries))
