from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.linker import EntityLinker
from medical_kg_nlp.linking.reranker import HeuristicReranker
from medical_kg_nlp.retrieval.bm25_retriever import BM25Retriever
from medical_kg_nlp.retrieval.candidate_generator import CandidateGenerator
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType


def test_bm25_uses_fixed_calibration_instead_of_query_maximum() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    candidates = BM25Retriever(store).retrieve("mơ hồ viêm", EntityType.DISEASE)

    assert candidates
    assert 0.0 < candidates[0].score < 1.0


def test_candidate_merge_is_order_independent_and_deduplicates_output_code() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = CandidateGenerator(store)
    exact = _candidate("concept-exact", "I10", 1.0, "exact")
    duplicate_code = _candidate("concept-fuzzy", "I10", 0.8, "fuzzy")
    unrelated = _candidate("concept-other", "I11", 0.7, "fuzzy")

    forward = sorted(
        generator._merge([exact, duplicate_code, unrelated]), key=lambda item: item.code or ""
    )
    reverse = sorted(
        generator._merge([unrelated, duplicate_code, exact]),
        key=lambda item: item.code or "",
    )

    assert [(item.code, item.score, item.sources) for item in forward] == [
        (item.code, item.score, item.sources) for item in reverse
    ]
    i10 = next(item for item in forward if item.code == "I10")
    assert i10.concept_id == "concept-exact"
    assert i10.sources == ("exact", "fuzzy")


def test_reranker_penalizes_conflicting_rxnorm_strength() -> None:
    entries = [
        _drug_entry("RX:1", "1", "Drug 1 mg tablet"),
        _drug_entry("RX:05", "05", "Drug 0.5 mg tablet"),
    ]
    reranker = HeuristicReranker(DictionaryStore(entries))

    ranked = reranker.rerank(
        [
            _candidate("RX:05", "05", 0.8, "exact"),
            _candidate("RX:1", "1", 0.8, "exact"),
        ],
        context_window="Dùng Drug 1 mg tablet mỗi ngày.",
        mention="Drug 1 mg tablet",
    )

    assert ranked[0].code == "1"
    assert ranked[0].score > ranked[1].score
    assert ranked[1].score <= 0.2


def test_linker_keeps_candidates_but_abstains_without_score_margin() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    linker = EntityLinker(
        CandidateGenerator(store),
        assignment_threshold=0.75,
        assignment_margin=0.05,
    )
    entity = EntityAnnotation(
        id="E1",
        span=(0, 4),
        text="test",
        normalized_text="test",
        type=EntityType.DISEASE,
        assertion=AssertionStatus.UNKNOWN,
    )

    linker.apply_candidates(
        entity,
        [
            _candidate("C1", "I10", 0.80, "exact"),
            _candidate("C2", "I11", 0.78, "fuzzy"),
        ],
    )

    assert entity.code is None
    assert entity.code_system == CodeSystem.NONE
    assert len(entity.candidates) == 2


def _candidate(concept_id: str, code: str, score: float, source: str) -> Candidate:
    return Candidate(
        concept_id=concept_id,
        code=code,
        code_system=CodeSystem.ICD10 if code.startswith("I") else CodeSystem.RXNORM,
        canonical_name=concept_id,
        semantic_type=EntityType.DISEASE if code.startswith("I") else EntityType.DRUG,
        score=score,
        source=source,
    )


def _drug_entry(concept_id: str, code: str, canonical_name: str) -> ConceptEntry:
    return ConceptEntry(
        concept_id=concept_id,
        code=code,
        code_system=CodeSystem.RXNORM,
        canonical_name=canonical_name,
        semantic_type=EntityType.DRUG,
        ingredient="drug",
        dose_form="tablet",
    )
