"""Fast model-adapter contracts that do not require optional ML dependencies."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from medical_kg_nlp.adapters import (
    HuggingFaceCrossEncoderAdapter,
    HuggingFaceModelConfig,
    HuggingFaceTextEncoderAdapter,
    HuggingFaceTokenClassifierAdapter,
    HybridEntityExtractorAdapter,
    MedicationMentionEntityExtractorAdapter,
    OptionalModelDependencyError,
)
from medical_kg_nlp.adapters.huggingface import runtime as huggingface_runtime
from medical_kg_nlp.adapters.model_spans import TokenPrediction, project_bio_predictions
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.pipeline import PipelineFactory, PipelineFactoryConfig, PipelineModelConfig
from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.retrieval import DenseHit, DenseRetrieverAdapter
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.memory import InMemoryTerminologyRepository


def test_model_config_requires_pinned_identity() -> None:
    with pytest.raises(ValueError, match="revision is required"):
        PipelineModelConfig.from_mapping(
            {"entity_extractor": {"model_id": "local/model"}}
        )

    config = PipelineModelConfig.from_mapping(
        {
            "entity_extractor": {
                "model_id": "local/model",
                "revision": "abc123",
                "batch_size": 4,
                "max_length": 128,
                "label_map": {"PROBLEM": "DISEASE"},
                "default_confidence_threshold": 0.4,
                "confidence_thresholds": {"SYMPTOM": 0.85, "LAB_RESULT": 0.9},
            }
        }
    )

    assert config.entity_extractor is not None
    assert config.entity_extractor.provenance == "local/model@abc123"
    assert config.entity_label_map == (("PROBLEM", EntityType.DISEASE),)
    assert config.entity_default_confidence_threshold == 0.4
    assert config.entity_confidence_thresholds == (
        (EntityType.LAB_RESULT, 0.9),
        (EntityType.SYMPTOM, 0.85),
    )
    assert config.entity_combine_with_dictionary is False

    with pytest.raises(ValueError, match="unknown entity type"):
        PipelineModelConfig.from_mapping(
            {
                "entity_extractor": {
                    "model_id": "local/model",
                    "revision": "abc123",
                    "confidence_thresholds": {"NOT_A_TYPE": 0.5},
                }
            }
        )


def test_model_adapter_construction_does_not_import_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail_import() -> tuple[Any, Any]:
        nonlocal calls
        calls += 1
        raise OptionalModelDependencyError("missing ml extra")

    monkeypatch.setattr(huggingface_runtime, "_import_model_dependencies", fail_import)
    adapter = HuggingFaceTextEncoderAdapter(_model_config())

    assert calls == 0
    with pytest.raises(OptionalModelDependencyError, match="missing ml extra"):
        adapter.encode(("metformin",))
    assert calls == 1


def test_core_pipeline_import_does_not_import_model_frameworks() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import medical_kg_nlp.pipeline; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr


def test_bio_projection_deduplicates_windows_and_preserves_raw_slice() -> None:
    text = "đau ngực và sốt"
    entities = project_bio_predictions(
        text,
        (
            TokenPrediction(0, 3, "B-SYMPTOM", 0.91),
            TokenPrediction(4, 8, "I-SYMPTOM", 0.93),
            TokenPrediction(4, 8, "I-SYMPTOM", 0.70),
            TokenPrediction(12, 15, "S-SYMPTOM", 0.88),
            TokenPrediction(100, 105, "B-DISEASE", 0.99),
        ),
    )

    assert [(text[start:end], item.entity_type) for item in entities for start, end in [item.span]] == [
        ("đau ngực", EntityType.SYMPTOM),
        ("sốt", EntityType.SYMPTOM),
    ]


def test_token_classifier_projects_fast_tokenizer_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HuggingFaceTokenClassifierAdapter(_model_config(), stride=8)
    tokenizer = _FakeTokenizer()
    model = _FakeTokenModel()
    monkeypatch.setattr(adapter, "_runtime", lambda: (_FakeTorch(), tokenizer, model))

    entities = adapter.extract("đau ngực")

    assert [(entity.text, entity.span, entity.type) for entity in entities] == [
        ("đau ngực", (0, 8), EntityType.SYMPTOM)
    ]
    entities[0].validate_offsets("đau ngực")


def test_token_classifier_applies_per_type_threshold_before_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HuggingFaceTokenClassifierAdapter(
        _model_config(),
        stride=8,
        confidence_thresholds={EntityType.SYMPTOM: 0.95},
    )
    monkeypatch.setattr(
        adapter,
        "_runtime",
        lambda: (_FakeTorch(), _FakeTokenizer(), _FakeTokenModel()),
    )

    assert adapter.extract("đau ngực") == []


def test_cross_encoder_reranks_without_changing_candidate_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HuggingFaceCrossEncoderAdapter(_model_config(), model_weight=1.0)
    candidates = [_candidate("A", 0.9), _candidate("B", 0.2)]
    monkeypatch.setattr(adapter, "_score_pairs", lambda pairs: [0.1, 0.95])

    reranked = adapter.rerank(candidates, "context", "mention")

    assert [candidate.concept_id for candidate in reranked] == ["B", "A"]
    assert candidates[0].score == 0.9


def test_dense_retriever_enforces_repository_type_constraints() -> None:
    drug = _entry("RX:6809", "6809", CodeSystem.RXNORM, EntityType.DRUG)
    disease = _entry("ICD:E11.9", "E11.9", CodeSystem.ICD10, EntityType.DISEASE)
    repository = InMemoryTerminologyRepository(DictionaryStore([drug, disease]))
    adapter = DenseRetrieverAdapter(
        encoder=_FakeEncoder(),
        index=_FakeDenseIndex(),
        repository=repository,
    )

    candidates = adapter.retrieve("metformin", EntityType.DRUG, "", 5)

    assert [candidate.concept_id for candidate in candidates] == ["RX:6809"]


def test_factory_wires_model_extractor_without_loading_weights(tmp_path: Path) -> None:
    dictionary = tmp_path / "concepts.jsonl"
    dictionary.write_text(
        '{"concept_id":"D1","code":"I10","code_system":"ICD-10",'
        '"canonical_name":"tăng huyết áp","semantic_type":"DISEASE"}\n',
        encoding="utf-8",
    )
    config = PipelineFactoryConfig(
        recognition_dictionary_path=str(dictionary),
        options=PipelineOptions(
            enable_context=False,
            enable_linking=False,
            enable_entity_kg_validation=False,
            enable_relations=False,
            enable_relation_kg_validation=False,
        ),
        models=PipelineModelConfig(entity_extractor=_model_config(), entity_stride=8),
    )

    runner = PipelineFactory.from_config(config)

    extractor = runner.components.entity_extractor
    assert isinstance(extractor, MedicationMentionEntityExtractorAdapter)
    assert isinstance(extractor.extractor, HuggingFaceTokenClassifierAdapter)


def test_factory_can_combine_model_with_reviewed_dictionary(tmp_path: Path) -> None:
    dictionary = tmp_path / "concepts.jsonl"
    dictionary.write_text(
        '{"concept_id":"D1","code":"I10","code_system":"ICD-10",'
        '"canonical_name":"tăng huyết áp","semantic_type":"DISEASE"}\n',
        encoding="utf-8",
    )
    config = PipelineFactoryConfig(
        recognition_dictionary_path=str(dictionary),
        options=PipelineOptions(
            enable_context=False,
            enable_linking=False,
            enable_entity_kg_validation=False,
            enable_relations=False,
            enable_relation_kg_validation=False,
        ),
        models=PipelineModelConfig(
            entity_extractor=_model_config(),
            entity_stride=8,
            entity_combine_with_dictionary=True,
        ),
    )

    runner = PipelineFactory.from_config(config)

    extractor = runner.components.entity_extractor
    assert isinstance(extractor, HybridEntityExtractorAdapter)
    assert isinstance(extractor.model, MedicationMentionEntityExtractorAdapter)


def test_factory_wires_cross_encoder_without_loading_weights(tmp_path: Path) -> None:
    dictionary = tmp_path / "concepts.jsonl"
    dictionary.write_text(
        '{"concept_id":"D1","code":"I10","code_system":"ICD-10",'
        '"canonical_name":"tăng huyết áp","semantic_type":"DISEASE"}\n',
        encoding="utf-8",
    )
    config = PipelineFactoryConfig(
        recognition_dictionary_path=str(dictionary),
        options=PipelineOptions(
            candidate_sources=("exact",),
            enable_context=False,
            enable_entity_kg_validation=False,
            enable_relations=False,
            enable_relation_kg_validation=False,
        ),
        models=PipelineModelConfig(candidate_reranker=_model_config()),
    )

    runner = PipelineFactory.from_config(config)

    assert isinstance(runner.components.candidate_reranker, HuggingFaceCrossEncoderAdapter)


def _model_config() -> HuggingFaceModelConfig:
    return HuggingFaceModelConfig(
        model_id="local/test-model",
        revision="deadbeef",
        batch_size=2,
        max_length=32,
    )


def _entry(
    concept_id: str,
    code: str,
    code_system: CodeSystem,
    entity_type: EntityType,
) -> ConceptEntry:
    return ConceptEntry(
        concept_id=concept_id,
        code=code,
        code_system=code_system,
        canonical_name=concept_id,
        semantic_type=entity_type,
    )


def _candidate(concept_id: str, score: float) -> Candidate:
    return Candidate(
        concept_id=concept_id,
        code=concept_id,
        code_system=CodeSystem.ICD10,
        canonical_name=concept_id,
        semantic_type=EntityType.DISEASE,
        score=score,
        source="exact",
    )


class _FakeTensor:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __getitem__(self, item: object) -> "_FakeTensor":
        return _FakeTensor(self.value[item])

    def to(self, device: str) -> "_FakeTensor":
        del device
        return self

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def tolist(self) -> Any:
        return self.value


class _FakeProbabilities:
    def max(self, dim: int) -> tuple[_FakeTensor, _FakeTensor]:
        assert dim == -1
        return _FakeTensor([[1.0, 0.95, 0.93, 1.0]]), _FakeTensor([[0, 1, 2, 0]])


class _FakeTorch:
    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()

    @staticmethod
    def softmax(logits: object, dim: int) -> _FakeProbabilities:
        del logits
        assert dim == -1
        return _FakeProbabilities()


class _FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, _FakeTensor]:
        del text, kwargs
        return {
            "input_ids": _FakeTensor([[0, 1, 2, 0]]),
            "attention_mask": _FakeTensor([[1, 1, 1, 1]]),
            "offset_mapping": _FakeTensor([[[0, 0], [0, 3], [4, 8], [0, 0]]]),
            "overflow_to_sample_mapping": _FakeTensor([0]),
        }


class _FakeTokenModel:
    config = SimpleNamespace(id2label={0: "O", 1: "B-SYMPTOM", 2: "I-SYMPTOM"})

    def __call__(self, **inputs: object) -> SimpleNamespace:
        del inputs
        return SimpleNamespace(logits=object())


class _FakeEncoder:
    def encode(self, texts: Any) -> list[tuple[float, ...]]:
        assert tuple(texts) == ("metformin",)
        return [(1.0, 0.0)]


class _FakeDenseIndex:
    def search(self, vector: Any, **kwargs: object) -> list[DenseHit]:
        assert tuple(vector) == (1.0, 0.0)
        assert kwargs["entity_type"] == EntityType.DRUG
        return [DenseHit("ICD:E11.9", 0.99), DenseHit("RX:6809", 0.95)]
