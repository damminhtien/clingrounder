from __future__ import annotations

from medical_kg_nlp.adapters import HuggingFaceCrossEncoderAdapter, HuggingFaceModelConfig
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.pipeline import (
    CandidateRerankRequest,
    PipelineComponents,
    PipelineOptions,
    PipelineRunner,
)
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_cross_encoder_batch_output_matches_scalar_output(monkeypatch) -> None:
    adapter = HuggingFaceCrossEncoderAdapter(
        _model_config(),
        model_weight=1.0,
        max_pairs_per_batch=2,
    )
    candidates = [
        _candidate("A", 0.2, "alpha"),
        _candidate("B", 0.3, "beta"),
    ]
    second_candidates = (candidates[0], _candidate("C", 0.1, "gamma"))

    def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
        return [0.9 if right in {"alpha", "gamma"} else 0.1 for _, right in pairs]

    monkeypatch.setattr(adapter, "_score_pairs", score_pairs)
    scalar = [
        adapter.rerank(request_candidates, "context", mention)
        for mention, request_candidates in (("one", candidates), ("two", second_candidates))
    ]
    batched = adapter.rerank_batch(
        (
            CandidateRerankRequest("one", "one", "context", tuple(candidates)),
            CandidateRerankRequest(
                "two",
                "two",
                "context",
                second_candidates,
            ),
        )
    )

    assert [item.concept_id for item in batched["one"]] == [
        item.concept_id for item in scalar[0]
    ]
    assert [item.concept_id for item in batched["two"]] == [
        item.concept_id for item in scalar[1]
    ]
    assert batched["one"] == scalar[0]


def test_cross_encoder_batch_bounds_pairs_and_token_estimate() -> None:
    adapter = HuggingFaceCrossEncoderAdapter(
        _model_config(),
        max_pairs_per_batch=2,
        max_tokens=8,
    )

    batches = adapter._bounded_pair_batches(
        [("one two", "a"), ("three", "b"), ("four", "c"), ("five", "d")]
    )

    assert [len(batch) for batch in batches] == [1, 1, 1, 1]


def test_batch_requests_are_typed_and_preserve_candidate_evidence() -> None:
    candidate = _candidate("A", 0.9, "alpha")
    request = CandidateRerankRequest("entity", "mention", "context", (candidate,))

    assert request.entity_id == "entity"
    assert request.candidates[0] is candidate
    assert request.candidates[0].evidence == candidate.evidence


def test_model_config_exposes_bounded_pair_settings() -> None:
    config = HuggingFaceModelConfig.from_mapping(
        {
            "model_id": "local/model",
            "revision": "revision",
            "max_pairs_per_batch": 4,
            "max_tokens": 128,
        },
        name="candidate_reranker",
    )

    assert config.max_pairs_per_batch == 4
    assert config.max_tokens == 128


def test_pipeline_prefers_batch_retriever_when_available() -> None:
    retriever = _BatchRetriever()
    runner = PipelineRunner(
        PipelineComponents(
            entity_extractor=_TwoEntityExtractor(),
            candidate_retriever=retriever,
            candidate_assigner=_NoOpAssigner(),
            options=PipelineOptions(
                enable_context=False,
                enable_candidate_reranking=False,
                enable_entity_kg_validation=False,
                enable_relations=False,
                enable_relation_kg_validation=False,
            ),
        )
    )

    runner.process_text("batch", "alpha beta")

    assert retriever.batch_calls == 1
    assert retriever.scalar_calls == 0


def _model_config() -> HuggingFaceModelConfig:
    return HuggingFaceModelConfig(model_id="local/model", revision="revision", batch_size=2)


def _candidate(concept_id: str, score: float, canonical_name: str) -> Candidate:
    return Candidate(
        concept_id=concept_id,
        code=concept_id,
        code_system=CodeSystem.ICD10,
        canonical_name=canonical_name,
        semantic_type=EntityType.DISEASE,
        score=score,
        source="test",
    )


class _BatchRetriever:
    def __init__(self) -> None:
        self.batch_calls = 0
        self.scalar_calls = 0

    def retrieve(self, entity, context_window="", mention=None):
        del entity, context_window, mention
        self.scalar_calls += 1
        raise AssertionError("scalar retriever should not be used")

    def retrieve_batch(self, requests):
        self.batch_calls += 1
        return {request.entity_id: [] for request in requests}


class _TwoEntityExtractor:
    def extract(self, source_text: str) -> list[EntityAnnotation]:
        return [
            EntityAnnotation("E1", (0, 5), source_text[:5], "alpha", EntityType.DISEASE),
            EntityAnnotation("E2", (6, 10), source_text[6:], "beta", EntityType.DISEASE),
        ]


class _NoOpAssigner:
    def assign(self, entity, candidates, *, mention=None):
        del candidates, mention
        return entity
