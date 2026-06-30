from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.retrieval.candidate_generator import CandidateGenerator
from medical_kg_nlp.retrieval.ngram_retriever import CharNgramRetriever
from medical_kg_nlp.schema.types import EntityType


def test_candidate_generation_handles_vietnamese_alias() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = CandidateGenerator(store, "data/dictionaries/abbreviations.jsonl")
    candidates = generator.generate("đái tháo đường type 2", EntityType.DISEASE)
    assert candidates[0].code == "E11"


def test_candidate_generation_handles_abbreviation() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = CandidateGenerator(store, "data/dictionaries/abbreviations.jsonl")
    candidates = generator.generate("T2DM", EntityType.DISEASE)
    assert any(candidate.code == "E11" for candidate in candidates)


def test_candidate_generation_handles_vietnamese_abbreviation() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = CandidateGenerator(store, "data/dictionaries/abbreviations.jsonl")
    candidates = generator.generate("THA", EntityType.DISEASE)
    assert candidates[0].code == "I10"


def test_char_ngram_retriever_handles_noisy_surface_form() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    candidates = CharNgramRetriever(store).retrieve("pnuemonia", EntityType.DISEASE)
    assert candidates[0].code == "J18.9"
    assert candidates[0].source == "char_ngram"


def test_candidate_generation_can_disable_char_ngram_source() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = CandidateGenerator(
        store,
        "data/dictionaries/abbreviations.jsonl",
        retrieval_sources=("exact", "abbreviation"),
    )
    candidates = generator.generate("pnuemonia", EntityType.DISEASE)
    assert candidates == []


def test_candidate_generation_rejects_unknown_retrieval_source() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    try:
        CandidateGenerator(store, retrieval_sources=("dense",))
    except ValueError as error:
        assert "dense" in str(error)
    else:
        raise AssertionError("Expected unknown retrieval source to fail.")
