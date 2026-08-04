from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.linking.learned_edits import (
    LearnedEditObservation,
    learn_edit_transformations,
)
from medical_kg_nlp.linking.mention_code_memory import (
    MentionCodeMemoryObservation,
    build_mention_code_memory,
)
from medical_kg_nlp.retrieval.dense_retriever import DenseHit, DenseRetrieverAdapter
from medical_kg_nlp.retrieval.rule_factory import build_in_memory_retrieval_pipeline as _retrieval
from medical_kg_nlp.retrieval.ngram_retriever import CharNgramRetriever
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.schema.types import CodeSystem
from medical_kg_nlp.terminology.memory import InMemoryTerminologyRepository


def test_candidate_generation_handles_vietnamese_alias() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")
    candidates = generator.retrieve("đái tháo đường type 2", EntityType.DISEASE)
    assert candidates[0].code == "E11"


def test_candidate_generation_handles_abbreviation() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")
    candidates = generator.retrieve("T2DM", EntityType.DISEASE)
    assert any(candidate.code == "E11" for candidate in candidates)


def test_candidate_generation_handles_vietnamese_abbreviation() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")
    candidates = generator.retrieve("THA", EntityType.DISEASE)
    assert candidates[0].code == "I10"


def test_char_ngram_retriever_handles_noisy_surface_form() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    candidates = CharNgramRetriever(store).retrieve("pnuemonia", EntityType.DISEASE)
    assert candidates[0].code == "J18.9"
    assert candidates[0].source == "char_ngram"


def test_candidate_generation_can_disable_char_ngram_source() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(
        store,
        "data/dictionaries/abbreviations.jsonl",
        retrieval_sources=("exact", "abbreviation"),
    )
    candidates = generator.retrieve("pnuemonia", EntityType.DISEASE)
    assert candidates == []


def test_candidate_generation_builds_only_enabled_approximate_indexes() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, retrieval_sources=("exact",))

    assert generator.retrieval_sources == ("exact",)


def test_candidate_generation_rejects_unknown_retrieval_source() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    try:
        _retrieval(store, retrieval_sources=("dense",))
    except ValueError as error:
        assert "dense" in str(error)
    else:
        raise AssertionError("Expected unknown retrieval source to fail.")


def test_learned_retrieval_sources_follow_precision_first_order() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    repository = InMemoryTerminologyRepository(store)
    memory = build_mention_code_memory(
        tuple(
            MentionCodeMemoryObservation(
                document_id=str(index),
                mention="THA",
                entity_type=EntityType.DISEASE,
                genre="unknown",
                code_system=CodeSystem.ICD10,
                code="I10",
            )
            for index in range(3)
        )
    )
    edits = learn_edit_transformations(
        tuple(
            LearnedEditObservation("đtđ", "đái tháo đường type 2", EntityType.DISEASE)
            for _ in range(3)
        )
    )
    dense = DenseRetrieverAdapter(_Encoder(), _DenseIndex(), repository)

    pipeline = _retrieval(
        store,
        "data/dictionaries/abbreviations.jsonl",
        retrieval_sources=(
            "mention_memory",
            "exact",
            "abbreviation",
            "learned_edit",
            "bm25",
            "dense",
        ),
        mention_code_memory=memory,
        learned_edit_model=edits,
        dense_retriever=dense,
    )

    assert [adapter.source for adapter in pipeline.retrievers] == [
        "reviewed_memory",
        "mention_memory",
        "exact",
        "abbreviation",
        "learned_edit",
        "mention_memory",
        "bm25",
        "dense",
    ]


class _Encoder:
    def encode(self, texts: tuple[str, ...]) -> list[tuple[float, ...]]:
        return [(1.0, 0.0) for _ in texts]


class _DenseIndex:
    def search(
        self,
        vector: tuple[float, ...],
        *,
        entity_type: EntityType,
        code_systems: tuple[CodeSystem, ...] | None,
        limit: int,
    ) -> list[DenseHit]:
        del vector, entity_type, code_systems, limit
        return []
