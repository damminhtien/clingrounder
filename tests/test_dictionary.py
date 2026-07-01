import json
import subprocess
import sys

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


def test_structured_icd_fields_expand_all_names() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entry = store.by_concept_id["ICD10:E11"]

    assert entry.official_name_vi == "Đái tháo đường type 2"
    assert entry.parent_code == "E10-E14"
    assert "T2DM" in entry.all_names
    assert "đái tháo đường týp II" in entry.all_names


def test_vietnamese_medical_alias_maps_to_icd_code() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = CandidateGenerator(store, "data/dictionaries/abbreviations.jsonl")

    hypertension = generator.generate("cao huyết áp", EntityType.DISEASE)
    myocardial_infarction = generator.generate("nhồi máu cơ tim", EntityType.DISEASE)
    copd = generator.generate("bệnh phổi tắc nghẽn mạn tính", EntityType.DISEASE)
    ckd = generator.generate("suy thận mạn", EntityType.DISEASE)
    gerd = generator.generate("GERD", EntityType.DISEASE)

    assert hypertension[0].code == "I10"
    assert myocardial_infarction[0].code == "I21.9"
    assert copd[0].code == "J44.9"
    assert ckd[0].code == "N18.9"
    assert gerd[0].code == "K21.9"


def test_rxnorm_drug_fields_expand_aliases_without_icd_leakage() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = CandidateGenerator(store, "data/dictionaries/abbreviations.jsonl")

    candidates = generator.generate("Ventolin", EntityType.DRUG)

    assert candidates[0].code == "435"
    assert candidates[0].code_system == CodeSystem.RXNORM
    assert all(candidate.code_system != CodeSystem.ICD10 for candidate in candidates)


def test_source_backed_rxnorm_terms_are_dictionary_constrained() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = CandidateGenerator(store, "data/dictionaries/abbreviations.jsonl")

    aspirin = generator.generate("ASA", EntityType.DRUG)
    lisinopril = generator.generate("Zestril", EntityType.DRUG)
    omeprazole = generator.generate("Prilosec", EntityType.DRUG)

    assert aspirin[0].code == "1191"
    assert lisinopril[0].code == "29046"
    assert omeprazole[0].code == "7646"
    assert all(candidate.code_system == CodeSystem.RXNORM for candidate in aspirin)


def test_blocked_alias_removes_false_positive_term() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")

    assert store.exact_lookup("hen") == []
    assert store.exact_lookup("hen phế quản")[0].code == "J45"


def test_build_dictionaries_validates_vietnamese_alias_table() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_dictionaries.py", "--config", "configs/default.yaml"],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)

    assert summary["concepts"] >= 31
    assert summary["assertion_cues"] >= 70
    assert summary["source_registry_entries"] >= 13
    assert summary["vietnamese_aliases"] == 16
