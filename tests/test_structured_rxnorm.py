from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.linker import EntityLinker
from medical_kg_nlp.linking.structured_rxnorm import (
    parse_medication_structure,
    rxnorm_structure_conflict,
)
from medical_kg_nlp.retrieval.candidate_generator import CandidateGenerator
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_rxnorm_structure_detects_strength_release_and_form_conflicts() -> None:
    entry = _entry()

    assert rxnorm_structure_conflict("metoprolol 25 mg oral tablet", entry) is None
    assert rxnorm_structure_conflict("metoprolol 50 mg oral tablet", entry) == (
        "rxnorm_product_strength_mismatch"
    )
    assert rxnorm_structure_conflict("metoprolol 25 mg IR tablet", entry) == (
        "rxnorm_release_mismatch"
    )
    # Route and dosage form are separate concepts; IV alone does not prove an injection form.
    assert rxnorm_structure_conflict("metoprolol 25 mg IV", entry) is None
    assert rxnorm_structure_conflict("metoprolol 25 mg XR tablet", entry) is None


def test_rxnorm_strength_normalization_handles_decimal_variants() -> None:
    comma = parse_medication_structure("digoxin 0,5 mg")
    dot = parse_medication_structure("digoxin .5 mg")

    assert comma.strengths == frozenset({"0.5mg"})
    assert dot.strengths == frozenset({"0.5mg"})


def test_administered_or_ambiguous_dose_does_not_hard_reject_product_strength() -> None:
    entry = _entry()

    assert rxnorm_structure_conflict("received metoprolol 50 mg IV", entry) is None
    assert rxnorm_structure_conflict("metoprolol 1.5 mg po qhs", entry) is None


def test_linker_does_not_treat_administered_dose_as_product_strength() -> None:
    entry = _entry()
    linker = EntityLinker(
        CandidateGenerator(DictionaryStore([entry])),
        candidate_threshold=0.5,
        emit_probabilities_by_source={"exact": 0.9},
    )
    entity = EntityAnnotation(
        id="E1",
        span=(0, 10),
        text="metoprolol",
        normalized_text="metoprolol",
        type=EntityType.DRUG,
    )
    candidate = Candidate(
        concept_id=entry.concept_id,
        code=entry.code,
        code_system=entry.code_system,
        canonical_name=entry.canonical_name,
        semantic_type=entry.semantic_type,
        score=0.99,
        source="exact",
        matched_alias="metoprolol",
    )

    linker.apply_candidates(entity, [candidate], mention="received metoprolol 50 mg IV")

    assert entity.code == "123"
    assert len(entity.candidates) == 1
    assert entity.candidates[0].qualified is True
    assert entity.candidates[0].qualification_reason == "qualified"
    assert entity.candidates[0].emit_probability == 0.9


def test_administered_dose_does_not_penalize_structured_product_candidate() -> None:
    entry = _entry()
    linker = EntityLinker(
        CandidateGenerator(DictionaryStore([entry])),
        candidate_threshold=0.9,
        emit_probabilities_by_source={"exact": 0.9},
    )
    entity = EntityAnnotation(
        id="E1",
        span=(0, 10),
        text="metoprolol",
        normalized_text="metoprolol",
        type=EntityType.DRUG,
    )
    candidate = Candidate(
        concept_id=entry.concept_id,
        code=entry.code,
        code_system=entry.code_system,
        canonical_name=entry.canonical_name,
        semantic_type=entry.semantic_type,
        score=0.9,
        source="exact",
        matched_alias="metoprolol",
    )

    linker.apply_candidates(entity, [candidate], mention="metoprolol 50 mg po qhs")

    assert entity.candidates[0].qualified is True
    assert entity.candidates[0].qualification_reason == "qualified"


def test_route_is_not_inferred_as_dose_form() -> None:
    oral = parse_medication_structure("amlodipine 5 mg po daily")
    intravenous = parse_medication_structure("methylprednisolone 125 mg IV")

    assert oral.routes == frozenset({"oral"})
    assert oral.dose_forms == frozenset()
    assert oral.administered_doses == frozenset({"5mg"})
    assert intravenous.routes == frozenset({"intravenous"})
    assert intravenous.dose_forms == frozenset()
    assert intravenous.administered_doses == frozenset({"125mg"})


def test_explicit_dose_form_separates_product_strength_from_administered_range() -> None:
    product = parse_medication_structure("clonazepam 0.5 mg oral tablet")
    range_dose = parse_medication_structure("acetaminophen 325-650 mg po q6h prn")

    assert product.product_strengths == frozenset({"0.5mg"})
    assert product.dose_forms == frozenset({"tablet"})
    assert range_dose.administered_doses == frozenset({"325mg"})
    assert not range_dose.product_strengths


def _entry() -> ConceptEntry:
    return ConceptEntry(
        concept_id="RXNORM:123",
        code="123",
        code_system=CodeSystem.RXNORM,
        canonical_name="metoprolol 25 MG Extended Release Oral Tablet",
        semantic_type=EntityType.DRUG,
        ingredient="metoprolol",
        dose_form="Oral Tablet",
        rxnorm_tty="SCD",
        strength="25 MG",
    )
