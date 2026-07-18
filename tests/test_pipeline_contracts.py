from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

from medical_kg_nlp.pipeline import (
    PipelineComponents,
    PipelineFactory,
    PipelineFactoryConfig,
    PipelineOptions,
    PipelineRunner,
)
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology import CachedTerminologyRepository


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
    config = PipelineFactoryConfig(
        options=PipelineOptions(enable_candidate_reranking=False)
    )

    assert config.options.enable_candidate_reranking is False


def test_factory_config_accepts_multiple_canonical_terminology_sources() -> None:
    config = PipelineFactoryConfig.from_mapping(
        {
            "terminology": {
                "normalization_paths": ["icd.jsonl", "rxnorm.jsonl"],
                "normalization_alias_overlay_paths": [
                    "mined-drug-aliases.jsonl",
                    "mined-disease-aliases.jsonl",
                ],
                "query_cache_size": 128,
                "reviewed_mention_path": "reviewed-memory.jsonl",
                "additional_recognition_paths": ["vn.jsonl", "mined.jsonl"],
            }
        }
    )

    assert config.normalization_dictionary_paths == ("icd.jsonl", "rxnorm.jsonl")
    assert config.normalization_alias_overlay_paths == (
        "mined-drug-aliases.jsonl",
        "mined-disease-aliases.jsonl",
    )
    assert config.terminology_query_cache_size == 128
    assert config.reviewed_mention_path == "reviewed-memory.jsonl"
    assert config.additional_recognition_dictionary_paths == ("vn.jsonl", "mined.jsonl")


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
        PipelineFactoryConfig.from_mapping(
            {"terminology": {"query_cache_size": -1}}
        )
