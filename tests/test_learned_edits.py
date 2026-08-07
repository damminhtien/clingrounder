"""Tests for supervised edit induction and dictionary-constrained expansion."""

from __future__ import annotations

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.dictionaries.synonym_table import ConceptEntry
from clingrounder.linking.learned_edits import (
    LearnedEditObservation,
    LearnedEditRetrieverAdapter,
    learn_edit_transformations,
    load_learned_edit_model,
    write_learned_edit_model,
)
from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.terminology.memory import InMemoryTerminologyRepository


def test_whole_abbreviation_activates_after_three_correct_observations() -> None:
    model = learn_edit_transformations(
        tuple(_observation("đtđ", "đái tháo đường") for _ in range(3))
    )

    variants = model.transform("ĐTĐ", EntityType.DISEASE)

    assert any(item.text == "đái tháo đường" for item in variants)
    assert all(rule.support >= 3 and rule.precision >= 0.9 for rule in model.rules)


def test_low_precision_short_edit_is_not_activated() -> None:
    model = learn_edit_transformations(
        (
            _observation("k phổi", "ung thư phổi"),
            _observation("k phổi", "ung thư phổi"),
            _observation("k phổi", "khám phổi"),
        )
    )

    assert model.transform("k phổi", EntityType.DISEASE) == ()


def test_token_edit_generalizes_inside_longer_mention() -> None:
    model = learn_edit_transformations(
        tuple(
            _observation(f"đtđ {suffix}", f"đái tháo đường {suffix}")
            for suffix in ("type 2", "biến chứng", "không biến chứng")
        )
    )

    variants = model.transform("đtđ mới phát hiện", EntityType.DISEASE)

    assert any(item.text == "đái tháo đường mới phát hiện" for item in variants)


def test_toneless_edit_restores_diacritics() -> None:
    model = learn_edit_transformations(
        tuple(_observation("tang huyet ap", "tăng huyết áp") for _ in range(3))
    )

    assert any(
        item.text == "tăng huyết áp"
        for item in model.transform("tang huyet ap", EntityType.DISEASE)
    )


def test_context_constrained_rule_does_not_execute_in_other_genre() -> None:
    model = learn_edit_transformations(
        tuple(
            _observation(
                "ha",
                "huyết áp",
                genre="lab_table",
                section="laboratory",
            )
            for _ in range(3)
        )
        + tuple(
            _observation("ha", "hạ", genre="clinical_note", section="symptom")
            for _ in range(2)
        )
    )

    assert any(
        item.text == "huyết áp"
        for item in model.transform(
            "ha",
            EntityType.DISEASE,
            genre="lab_table",
            section="laboratory",
        )
    )
    assert (
        model.transform(
            "ha",
            EntityType.DISEASE,
            genre="clinical_note",
            section="symptom",
        )
        == ()
    )


def test_retriever_resolves_only_type_compatible_dictionary_codes() -> None:
    disease = ConceptEntry(
        concept_id="ICD:E11.9",
        code="E11.9",
        code_system=CodeSystem.ICD10,
        canonical_name="đái tháo đường",
        semantic_type=EntityType.DISEASE,
    )
    drug = ConceptEntry(
        concept_id="RX:1",
        code="1",
        code_system=CodeSystem.RXNORM,
        canonical_name="đái tháo đường",
        semantic_type=EntityType.DRUG,
    )
    repository = InMemoryTerminologyRepository(DictionaryStore([disease, drug]))
    model = learn_edit_transformations(
        tuple(_observation("đtđ", "đái tháo đường") for _ in range(3))
    )
    retriever = LearnedEditRetrieverAdapter(model, repository)

    candidates = retriever.retrieve("đtđ", EntityType.DISEASE, "", 5)

    assert [item.code for item in candidates] == ["E11.9"]
    assert candidates[0].source == "learned_edit"


def test_learned_edit_model_round_trip(tmp_path) -> None:
    model = learn_edit_transformations(
        tuple(_observation("đtđ", "đái tháo đường") for _ in range(3))
    )
    path = tmp_path / "edits.jsonl"

    write_learned_edit_model(model, path)

    assert load_learned_edit_model(path) == model


def _observation(
    mention: str,
    alias: str,
    *,
    genre: str | None = None,
    section: str | None = None,
) -> LearnedEditObservation:
    return LearnedEditObservation(
        mention=mention,
        terminology_alias=alias,
        entity_type=EntityType.DISEASE,
        genre=genre,
        section=section,
    )
