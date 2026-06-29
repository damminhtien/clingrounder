from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.retrieval.candidate_generator import CandidateGenerator
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_drug_type_constraint_excludes_icd10() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = CandidateGenerator(store, "data/dictionaries/abbreviations.jsonl")
    candidates = generator.generate("metformin", EntityType.DRUG)
    assert candidates
    assert all(candidate.code_system == CodeSystem.RXNORM for candidate in candidates)


def test_disease_type_constraint_excludes_rxnorm() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = CandidateGenerator(store, "data/dictionaries/abbreviations.jsonl")
    candidates = generator.generate("type 2 diabetes", EntityType.DISEASE)
    assert candidates[0].code == "E11"
    assert all(candidate.code_system != CodeSystem.RXNORM for candidate in candidates)

