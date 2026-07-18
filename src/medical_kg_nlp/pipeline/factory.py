"""Composition root for concrete pipeline components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from medical_kg_nlp.adapters.rules import (
    DictionaryCandidateAdapter,
    KGValidatorAdapter,
    RuleAssertionClassifierAdapter,
    RuleEntityExtractorAdapter,
    RuleRelationExtractorAdapter,
)
from medical_kg_nlp.adapters.huggingface import (
    HuggingFaceCrossEncoderAdapter,
    HuggingFaceTokenClassifierAdapter,
)
from medical_kg_nlp.context.assertion import AssertionClassifier
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.merge import merge_concept_entries
from medical_kg_nlp.kg.validator import KGValidator
from medical_kg_nlp.linking.linker import EntityLinker
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.pipeline.components import PipelineComponents
from medical_kg_nlp.pipeline.model_config import PipelineModelConfig
from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.pipeline.ports import CandidateRerankerPort, EntityExtractorPort
from medical_kg_nlp.pipeline.runner import PipelineRunner
from medical_kg_nlp.preprocessing.normalizer import (
    DEFAULT_NORMALIZATION_CONTRACT,
    NormalizationContract,
)
from medical_kg_nlp.relations.rule_relations import RuleRelationExtractor
from medical_kg_nlp.retrieval.rule_factory import build_rule_retrieval_pipeline
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.terminology import (
    CompositeTerminologyRepository,
    InMemoryTerminologyRepository,
    SQLiteTerminologyRepository,
    TerminologyRepository,
    terminology_cache_path,
)

__all__ = ["PipelineFactory", "PipelineFactoryConfig"]


@dataclass(frozen=True)
class PipelineFactoryConfig:
    """Serializable configuration consumed by the composition root."""

    recognition_dictionary_path: str = "data/dictionaries/seed_concepts.jsonl"
    normalization_dictionary_paths: tuple[str, ...] = ()
    normalization_index_path: str | None = None
    normalization_alias_overlay_paths: tuple[str, ...] = ()
    terminology_cache_dir: str = ".cache/medical-kg/terminology"
    additional_recognition_dictionary_path: str | None = None
    abbreviation_path: str = "data/dictionaries/abbreviations.jsonl"
    alias_overlay_path: str | None = "data/dictionaries/vietnamese_medical_alias.jsonl"
    pipeline_version: str = "0.2.0"
    options: PipelineOptions = field(default_factory=PipelineOptions)
    models: PipelineModelConfig = field(default_factory=PipelineModelConfig)
    normalization_contract: NormalizationContract = DEFAULT_NORMALIZATION_CONTRACT

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PipelineFactoryConfig":
        terminology = _mapping(payload.get("terminology"), "terminology")
        pipeline = _mapping(payload.get("pipeline"), "pipeline")
        models = _mapping(payload.get("models"), "models")
        return cls(
            recognition_dictionary_path=_string(
                terminology,
                "recognition_path",
                cls.recognition_dictionary_path,
            ),
            normalization_dictionary_paths=_string_tuple(
                terminology.get("normalization_paths"),
                "normalization_paths",
            ),
            normalization_index_path=_optional_string(
                terminology.get("normalization_index_path")
            ),
            normalization_alias_overlay_paths=_string_tuple(
                terminology.get("normalization_alias_overlay_paths"),
                "normalization_alias_overlay_paths",
            ),
            terminology_cache_dir=_string(
                terminology,
                "cache_dir",
                cls.terminology_cache_dir,
            ),
            additional_recognition_dictionary_path=_optional_string(
                terminology.get("additional_recognition_path")
            ),
            abbreviation_path=_string(
                terminology,
                "abbreviation_path",
                cls.abbreviation_path,
            ),
            alias_overlay_path=_optional_string(
                terminology.get("alias_overlay_path", cls.alias_overlay_path)
            ),
            pipeline_version=_string(pipeline, "version", cls.pipeline_version),
            options=PipelineOptions.from_mapping(pipeline),
            models=PipelineModelConfig.from_mapping(models),
        )


class PipelineFactory:
    """Build a runner from configuration without leaking IO into orchestration."""

    @classmethod
    def from_config(
        cls,
        config: PipelineFactoryConfig | Mapping[str, object] | None = None,
    ) -> PipelineRunner:
        resolved = cls._resolve(config)
        recognition_entries = DictionaryStore.load_entries_jsonl(
            resolved.recognition_dictionary_path,
            alias_overlay_path=resolved.alias_overlay_path,
        )
        if resolved.additional_recognition_dictionary_path is not None:
            recognition_entries.extend(
                DictionaryStore.load_entries_jsonl(
                    resolved.additional_recognition_dictionary_path
                )
            )
        recognition_store = DictionaryStore(merge_concept_entries(recognition_entries))

        recognition_repository = InMemoryTerminologyRepository(recognition_store)
        terminology_repository: TerminologyRepository = recognition_repository
        uses_sqlite_normalization = bool(resolved.normalization_dictionary_paths)
        if resolved.normalization_dictionary_paths:
            # SCALING: canonical releases remain separate immutable files; the composition root
            # validates one content-addressed SQLite index against the complete ordered source set.
            source_paths = resolved.normalization_dictionary_paths
            index_path = resolved.normalization_index_path or str(
                terminology_cache_path(
                    resolved.terminology_cache_dir,
                    source_paths,
                    alias_overlay_paths=resolved.normalization_alias_overlay_paths,
                )
            )
            sqlite_repository = SQLiteTerminologyRepository(
                index_path,
                expected_source_paths=source_paths,
                expected_alias_overlay_paths=resolved.normalization_alias_overlay_paths,
                expected_normalization_version=resolved.normalization_contract.version,
            )
            terminology_repository = CompositeTerminologyRepository(
                (recognition_repository, sqlite_repository)
            )

        options = resolved.options
        entity_extractor: EntityExtractorPort
        if resolved.models.entity_extractor is not None:
            entity_extractor = HuggingFaceTokenClassifierAdapter(
                resolved.models.entity_extractor,
                label_map=dict(resolved.models.entity_label_map),
                stride=resolved.models.entity_stride,
            )
        else:
            entity_extractor = RuleEntityExtractorAdapter(
                RuleBasedNER(
                    recognition_store,
                    emit_probabilities_by_source=dict(
                        options.link_emit_probabilities_by_source
                    ),
                )
            )
        assertion_classifier = (
            RuleAssertionClassifierAdapter(AssertionClassifier())
            if options.enable_context
            else None
        )

        candidate_adapter: DictionaryCandidateAdapter | None = None
        if options.enable_linking:
            linker = EntityLinker(
                build_rule_retrieval_pipeline(
                    terminology_repository,
                    approximate_store=recognition_store,
                    abbreviation_path=resolved.abbreviation_path,
                    max_candidates=options.max_candidates,
                    retrieval_sources=options.candidate_sources,
                    use_fts_for_bm25=uses_sqlite_normalization,
                ),
                terminology_repository,
                assignment_threshold=options.link_assignment_threshold,
                assignment_margin=options.link_assignment_margin,
                candidate_threshold=options.link_candidate_threshold,
                candidate_relative_margin=options.link_candidate_relative_margin,
                max_qualified_candidates=options.link_max_qualified_candidates,
                candidate_thresholds_by_entity_type={
                    EntityType(entity_type): threshold
                    for entity_type, threshold in options.link_candidate_thresholds_by_type
                },
                candidate_thresholds_by_source=dict(
                    options.link_candidate_thresholds_by_source
                ),
                emit_probabilities_by_source=dict(
                    options.link_emit_probabilities_by_source
                ),
                enforce_rxnorm_structure=options.link_enforce_rxnorm_structure,
            )
            candidate_adapter = DictionaryCandidateAdapter(linker)

        candidate_reranker: CandidateRerankerPort | None = candidate_adapter
        if options.enable_linking and resolved.models.candidate_reranker is not None:
            candidate_reranker = HuggingFaceCrossEncoderAdapter(
                resolved.models.candidate_reranker,
                model_weight=resolved.models.candidate_reranker_weight,
                positive_label_index=resolved.models.candidate_positive_label_index,
            )

        relation_extractor = (
            RuleRelationExtractorAdapter(RuleRelationExtractor())
            if options.enable_relations
            else None
        )
        knowledge_validator = (
            KGValidatorAdapter(
                KGValidator(
                    recognition_store
                    if options.enable_entity_kg_validation
                    or options.enable_relation_kg_validation
                    else None
                )
            )
            if options.enable_entity_kg_validation
            or options.enable_relation_kg_validation
            else None
        )
        components = PipelineComponents(
            entity_extractor=entity_extractor,
            assertion_classifier=assertion_classifier,
            candidate_retriever=candidate_adapter,
            candidate_reranker=candidate_reranker,
            candidate_assigner=candidate_adapter,
            relation_extractor=relation_extractor,
            knowledge_validator=knowledge_validator,
            terminology_repository=terminology_repository,
            options=options,
            normalization_contract=resolved.normalization_contract,
            pipeline_version=resolved.pipeline_version,
        )
        return PipelineRunner(components)

    @staticmethod
    def _resolve(
        config: PipelineFactoryConfig | Mapping[str, object] | None,
    ) -> PipelineFactoryConfig:
        if config is None:
            return PipelineFactoryConfig()
        if isinstance(config, PipelineFactoryConfig):
            return config
        return PipelineFactoryConfig.from_mapping(config)


def _mapping(value: object, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _string(payload: Mapping[str, object], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("optional path values must be non-empty strings")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be an array of non-empty strings")
    return tuple(value)
