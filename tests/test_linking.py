from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.linker import EntityLinker
from medical_kg_nlp.linking.reranker import HeuristicReranker
from medical_kg_nlp.retrieval.adapters import (
    ExactRetrieverAdapter,
    ReviewedMentionRetrieverAdapter,
)
from medical_kg_nlp.retrieval.bm25_retriever import BM25Retriever
from medical_kg_nlp.retrieval.pipeline import RetrievalPipeline
from medical_kg_nlp.retrieval.rule_factory import build_in_memory_retrieval_pipeline as _retrieval
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType
from medical_kg_nlp.terminology.memory import InMemoryTerminologyRepository


def test_bm25_uses_fixed_calibration_instead_of_query_maximum() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    candidates = BM25Retriever(store).retrieve("mơ hồ viêm", EntityType.DISEASE)

    assert candidates
    assert 0.0 < candidates[0].score < 1.0


def test_candidate_merge_is_order_independent_and_deduplicates_output_code() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store)
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


def test_unique_exact_output_short_circuits_approximate_retrieval() -> None:
    store = DictionaryStore(
        [
            ConceptEntry(
                concept_id="ICD10:I10",
                code="I10",
                code_system=CodeSystem.ICD10,
                canonical_name="tăng huyết áp",
                semantic_type=EntityType.DISEASE,
            )
        ]
    )
    repository = InMemoryTerminologyRepository(store)

    class FailingRetriever:
        source = "fuzzy"
        terminal_on_match = False
        unique_output_short_circuit = False

        def retrieve(
            self,
            mention: str,
            entity_type: EntityType,
            context_window: str,
            limit: int,
        ) -> list[Candidate]:
            raise AssertionError("approximate retriever should not run")

    generator = RetrievalPipeline(
        repository,
        (ExactRetrieverAdapter(repository), FailingRetriever()),
    )

    candidates = generator.retrieve("tăng huyết áp", EntityType.DISEASE)

    assert [(candidate.code, candidate.sources) for candidate in candidates] == [
        ("I10", ("exact",))
    ]
    assert candidates[0].score == 1.0


def test_reviewed_memory_short_circuits_conflicting_seed_exact_match(
    tmp_path: Path,
) -> None:
    store = DictionaryStore(
        [
            ConceptEntry(
                concept_id="ICD10:I25.10",
                code="I25.10",
                code_system=CodeSystem.ICD10,
                canonical_name="bệnh động mạch vành",
                semantic_type=EntityType.DISEASE,
            ),
            ConceptEntry(
                concept_id="ICD10:I25.1",
                code="I25.1",
                code_system=CodeSystem.ICD10,
                canonical_name="coronary artery disease",
                aliases=("bệnh động mạch vành",),
                semantic_type=EntityType.DISEASE,
            ),
        ]
    )
    repository = InMemoryTerminologyRepository(store)
    memory = tmp_path / "reviewed.jsonl"
    memory.write_text(
        '{"mention":"bệnh động mạch vành","code_system":"ICD-10",'
        '"code":"I25.1","provenance":"reviewed_memory:test"}\n',
        encoding="utf-8",
    )
    generator = RetrievalPipeline(
        repository,
        (
            ReviewedMentionRetrieverAdapter.from_jsonl(repository, memory),
            ExactRetrieverAdapter(repository),
        ),
    )

    candidates = generator.retrieve("bệnh động mạch vành", EntityType.DISEASE)

    assert [(candidate.code, candidate.source) for candidate in candidates] == [
        ("I25.1", "reviewed_memory:test")
    ]


def test_ambiguous_exact_output_continues_approximate_retrieval() -> None:
    store = DictionaryStore(
        [
            ConceptEntry(
                concept_id="RXNORM:IN",
                code="1",
                code_system=CodeSystem.RXNORM,
                canonical_name="test drug",
                semantic_type=EntityType.DRUG,
                synonyms=("brand",),
            ),
            ConceptEntry(
                concept_id="RXNORM:SCD",
                code="2",
                code_system=CodeSystem.RXNORM,
                canonical_name="test drug 5 mg tablet",
                semantic_type=EntityType.DRUG,
                synonyms=("brand",),
            ),
        ]
    )
    repository = InMemoryTerminologyRepository(store)
    calls: list[str] = []

    class RecordingRetriever:
        source = "fuzzy"
        terminal_on_match = False
        unique_output_short_circuit = False

        def retrieve(
            self,
            mention: str,
            entity_type: EntityType,
            context_window: str,
            limit: int,
        ) -> list[Candidate]:
            calls.append(mention)
            return []

    generator = RetrievalPipeline(
        repository,
        (ExactRetrieverAdapter(repository), RecordingRetriever()),
    )

    generator.retrieve("brand", EntityType.DRUG)

    assert calls == ["brand"]


def test_reranker_penalizes_conflicting_rxnorm_strength() -> None:
    entries = [
        _drug_entry("RX:1", "1", "Drug 1 mg tablet"),
        _drug_entry("RX:05", "05", "Drug 0.5 mg tablet"),
    ]
    reranker = HeuristicReranker(
        InMemoryTerminologyRepository(DictionaryStore(entries))
    )

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


def test_reranker_breaks_exact_score_tie_with_administered_dose_without_hard_reject() -> None:
    entries = [
        _drug_entry("RX:1", "1", "clonazepam 1 mg oral tablet"),
        _drug_entry("RX:2", "2", "clonazepam 2 mg oral tablet"),
    ]
    reranker = HeuristicReranker(
        InMemoryTerminologyRepository(DictionaryStore(entries))
    )

    ranked = reranker.rerank(
        [
            _candidate("RX:2", "2", 1.0, "exact"),
            _candidate("RX:1", "1", 1.0, "exact"),
        ],
        mention="clonazepam 1 mg po qhs",
    )

    assert ranked[0].code == "1"
    assert ranked[0].score == 1.0
    assert ranked[1].score == 0.85


def test_reranker_uses_rxnorm_tty_for_bare_vs_structured_drug_mentions() -> None:
    entries = [
        ConceptEntry(
            concept_id="RXNORM:1364430",
            code="1364430",
            code_system=CodeSystem.RXNORM,
            canonical_name="apixaban",
            semantic_type=EntityType.DRUG,
            aliases=("Eliquis",),
            ingredient="apixaban",
            brand_name="Eliquis",
            rxnorm_tty="IN",
        ),
        ConceptEntry(
            concept_id="RXNORM:1364445",
            code="1364445",
            code_system=CodeSystem.RXNORM,
            canonical_name="apixaban 5 MG Oral Tablet",
            semantic_type=EntityType.DRUG,
            ingredient="apixaban",
            brand_name="Eliquis",
            dose_form="Oral Tablet",
            rxnorm_tty="SCD",
            strength="5 MG",
        ),
    ]
    reranker = HeuristicReranker(
        InMemoryTerminologyRepository(DictionaryStore(entries))
    )
    candidates = [
        _candidate("RXNORM:1364430", "1364430", 0.8, "fuzzy"),
        _candidate("RXNORM:1364445", "1364445", 0.8, "fuzzy"),
    ]

    bare = reranker.rerank(candidates, mention="Eliquis")
    structured = reranker.rerank(candidates, mention="apixaban 5 mg oral tablet")

    assert bare[0].code == "1364430"
    assert structured[0].code == "1364445"
    assert bare[1].score < bare[0].score
    assert structured[1].score < structured[0].score


def test_linker_keeps_candidates_but_abstains_without_score_margin() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    linker = EntityLinker(
        _retrieval(store),
        InMemoryTerminologyRepository(store),
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
    assert [candidate.qualified for candidate in entity.candidates] == [True, True]


def test_linker_rejects_candidates_below_absolute_threshold() -> None:
    linker = _linker()
    entity = _entity()
    entity.code_system = CodeSystem.ICD10
    entity.code = "I99"

    linker.apply_candidates(entity, [_candidate("C1", "I10", 0.61, "fuzzy")])

    assert entity.code is None
    assert entity.code_system == CodeSystem.NONE
    assert entity.candidates[0].qualified is False
    assert entity.candidates[0].qualification_reason == "below_absolute_threshold"


def test_linker_qualifies_dynamic_top_k_within_relative_margin() -> None:
    linker = _linker()
    entity = _entity()

    linker.apply_candidates(
        entity,
        [
            _candidate("C1", "I10", 0.90, "exact"),
            _candidate("C2", "I11", 0.88, "fuzzy"),
            _candidate("C3", "I12", 0.80, "fuzzy"),
        ],
    )

    assert entity.code is None
    assert [candidate.qualified for candidate in entity.candidates] == [True, True, False]
    assert entity.candidates[2].qualification_reason == "outside_relative_margin"


def test_linker_caps_qualified_candidates_at_five() -> None:
    linker = _linker(candidate_relative_margin=0.10)
    entity = _entity()
    candidates = [
        _candidate(f"C{index}", f"I{index:02d}", 0.90 - index / 100, "fuzzy")
        for index in range(6)
    ]

    linker.apply_candidates(entity, candidates)

    assert sum(candidate.qualified for candidate in entity.candidates) == 5
    assert entity.candidates[5].qualification_reason == "beyond_max_candidates"


def test_linker_supports_type_and_source_specific_candidate_thresholds() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    linker = EntityLinker(
        _retrieval(store),
        InMemoryTerminologyRepository(store),
        candidate_threshold=0.75,
        candidate_thresholds_by_entity_type={EntityType.DISEASE: 0.85},
        candidate_thresholds_by_source={"exact": 0.70},
    )

    exact_entity = _entity()
    linker.apply_candidates(exact_entity, [_candidate("C1", "I10", 0.80, "exact")])
    fuzzy_entity = _entity()
    linker.apply_candidates(fuzzy_entity, [_candidate("C2", "I11", 0.80, "fuzzy")])

    assert exact_entity.candidates[0].qualified is True
    assert fuzzy_entity.candidates[0].qualified is False


def _linker(*, candidate_relative_margin: float = 0.05) -> EntityLinker:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    return EntityLinker(
        _retrieval(store),
        InMemoryTerminologyRepository(store),
        candidate_relative_margin=candidate_relative_margin,
    )


def _entity() -> EntityAnnotation:
    return EntityAnnotation(
        id="E1",
        span=(0, 4),
        text="test",
        normalized_text="test",
        type=EntityType.DISEASE,
        assertion=AssertionStatus.UNKNOWN,
    )


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
