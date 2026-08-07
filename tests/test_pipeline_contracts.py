from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.pipeline import (
    PipelineComponents,
    PipelineFactory,
    PipelineConfig,
    PipelineOptions,
    PipelineRunner,
)
from clingrounder.schema.annotation import EntityAnnotation
from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.terminology import (
    CachedTerminologyRepository,
    InMemoryTerminologyRepository,
)


@dataclass(frozen=True)
class EmptyEntityExtractorAdapter:
    def extract(self, source_text: str) -> list[EntityAnnotation]:
        return []


@dataclass(frozen=True)
class InvalidDrugCodeExtractorAdapter:
    """Return a type/code-system mismatch to exercise the runner's core gate."""

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        return [
            EntityAnnotation(
                id="e1",
                span=(0, len(source_text)),
                text=source_text,
                normalized_text=source_text.casefold(),
                type=EntityType.DRUG,
                code_system=CodeSystem.ICD10,
                code="I10",
                confidence=1.0,
            )
        ]


@dataclass(frozen=True)
class FabricatedDiseaseCodeExtractorAdapter:
    """Return a correctly typed code absent from the active release."""

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        return [
            EntityAnnotation(
                id="e1",
                span=(0, len(source_text)),
                text=source_text,
                normalized_text=source_text.casefold(),
                type=EntityType.DISEASE,
                code_system=CodeSystem.ICD10,
                code="ZZZ.999",
                confidence=1.0,
            )
        ]


def test_pipeline_runner_accepts_only_components() -> None:
    parameters = inspect.signature(PipelineRunner).parameters

    assert list(parameters) == ["components"]


def test_pipeline_runner_can_use_minimal_custom_components() -> None:
    options = PipelineOptions(
        enable_context=False,
        enable_linking=False,
        enable_candidate_reranking=False,
        enable_entity_kg_validation=False,
        enable_relations=False,
        enable_relation_kg_validation=False,
    )
    runner = PipelineRunner(
        PipelineComponents(
            entity_extractor=EmptyEntityExtractorAdapter(),
            options=options,
            pipeline_version="contract-test",
        )
    )

    prediction = runner.process_text("doc", "Không có entity.")

    assert prediction.entities == []
    assert prediction.metadata.pipeline_version == "contract-test"


def test_pipeline_runner_always_enforces_core_validation() -> None:
    options = PipelineOptions(
        enable_context=False,
        enable_linking=False,
        enable_candidate_reranking=False,
        enable_entity_kg_validation=False,
        enable_relations=False,
        enable_relation_kg_validation=False,
    )
    runner = PipelineRunner(
        PipelineComponents(
            entity_extractor=InvalidDrugCodeExtractorAdapter(),
            options=options,
        )
    )

    with pytest.raises(ValueError, match="invalid_code_system"):
        runner.process_text("doc", "aspirin")


def test_pipeline_runner_rejects_fabricated_correctly_typed_code() -> None:
    options = PipelineOptions(
        enable_context=False,
        enable_linking=False,
        enable_candidate_reranking=False,
        enable_entity_kg_validation=False,
        enable_relations=False,
        enable_relation_kg_validation=False,
    )
    terminology = InMemoryTerminologyRepository(
        DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    )
    runner = PipelineRunner(
        PipelineComponents(
            entity_extractor=FabricatedDiseaseCodeExtractorAdapter(),
            terminology_repository=terminology,
            options=options,
        )
    )

    with pytest.raises(ValueError, match="unknown_dictionary_code"):
        runner.process_text("doc", "bệnh giả")


def test_pipeline_components_reject_missing_enabled_port() -> None:
    with pytest.raises(ValueError, match="assertion_classifier"):
        PipelineComponents(entity_extractor=EmptyEntityExtractorAdapter())


def test_pipeline_factory_mapping_is_the_composition_root() -> None:
    runner = PipelineFactory.from_config(
        {
            "terminology": {
                "recognition_path": "data/dictionaries/seed_concepts.jsonl",
                "abbreviation_path": "data/dictionaries/abbreviations.jsonl",
                "alias_overlay_path": None,
            },
            "pipeline": {
                "version": "factory-test",
                "enable_relations": False,
                "enable_relation_kg_validation": False,
            },
        }
    )

    assert runner.components.pipeline_version == "factory-test"
    assert runner.components.relation_extractor is None
    assert runner.components.terminology_repository is not None


def test_factory_config_remains_serializable_for_process_workers() -> None:
    config = PipelineConfig(
        options=PipelineOptions(enable_candidate_reranking=False)
    )

    assert config.options.enable_candidate_reranking is False


def test_factory_config_accepts_multiple_canonical_terminology_sources() -> None:
    config = PipelineConfig.from_mapping(
        {
            "terminology": {
                "normalization_paths": ["icd.jsonl", "rxnorm.jsonl"],
                "normalization_alias_overlay_paths": [
                    "mined-drug-aliases.jsonl",
                    "mined-disease-aliases.jsonl",
                ],
                "query_cache_size": 128,
                "reviewed_mention_path": "reviewed-memory.jsonl",
                "mention_code_memory_path": "mention-code-memory.jsonl",
                "learned_edit_path": "learned-edits.jsonl",
                "synonym_index_path": "synonym-index",
                "synonym_index_terminology_fingerprint": "a" * 64,
                "additional_recognition_paths": ["vn.jsonl", "mined.jsonl"],
            },
            "models": {
                "candidate_dense_encoder": {
                    "model_id": "local/xlmr-linker",
                    "revision": "abc123",
                }
            },
        }
    )

    assert config.terminology.normalization_dictionary_paths == ("icd.jsonl", "rxnorm.jsonl")
    assert config.terminology.normalization_alias_overlay_paths == (
        "mined-drug-aliases.jsonl",
        "mined-disease-aliases.jsonl",
    )
    assert config.terminology.terminology_query_cache_size == 128
    assert config.terminology.reviewed_mention_path == "reviewed-memory.jsonl"
    assert config.terminology.mention_code_memory_path == "mention-code-memory.jsonl"
    assert config.terminology.learned_edit_path == "learned-edits.jsonl"
    assert config.terminology.synonym_index_path == "synonym-index"
    assert config.terminology.synonym_index_terminology_fingerprint == "a" * 64
    assert config.models.candidate_dense_encoder is not None
    assert config.models.candidate_dense_encoder.revision == "abc123"
    assert config.terminology.additional_recognition_dictionary_paths == (
        "vn.jsonl",
        "mined.jsonl",
    )


def test_pipeline_factory_wraps_terminology_with_bounded_cache() -> None:
    runner = PipelineFactory.from_config(
        {
            "terminology": {
                "recognition_path": "data/dictionaries/seed_concepts.jsonl",
                "alias_overlay_path": None,
                "query_cache_size": 8,
            },
            "pipeline": {
                "enable_context": False,
                "enable_linking": False,
                "enable_candidate_reranking": False,
                "enable_entity_kg_validation": False,
                "enable_relations": False,
                "enable_relation_kg_validation": False,
            },
        }
    )

    repository = runner.components.terminology_repository
    assert isinstance(repository, CachedTerminologyRepository)
    assert repository.cache_info().max_size == 8


def test_factory_config_rejects_negative_terminology_cache_size() -> None:
    with pytest.raises(ValueError, match="query_cache_size"):
        PipelineConfig.from_mapping(
            {"terminology": {"query_cache_size": -1}}
        )


def test_pipeline_options_parse_graph_evidence_second_pass() -> None:
    options = PipelineOptions.from_mapping(
        {
            "enable_graph_evidence_reranking": True,
            "graph_evidence_max_bonus": 0.03,
            "graph_evidence_min_support": 3,
            "graph_evidence_relation_types": ["CO_OCCURS_WITH", "TREATS"],
            "graph_evidence_cache_size": 128,
        }
    )

    assert options.enable_graph_evidence_reranking is True
    assert options.graph_evidence_max_bonus == 0.03
    assert options.graph_evidence_min_support == 3
    assert options.graph_evidence_relation_types == ("CO_OCCURS_WITH", "TREATS")
    assert options.graph_evidence_cache_size == 128


def test_graph_evidence_second_pass_requires_graph_index() -> None:
    with pytest.raises(ValueError, match="knowledge_graph_index_path"):
        PipelineFactory.from_config(
            PipelineConfig(
                options=PipelineOptions(enable_graph_evidence_reranking=True)
            )
        )
