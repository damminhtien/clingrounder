"""Composition root for concrete pipeline components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from medical_kg_nlp.adapters.rules import (
    KGValidatorAdapter,
    RuleAssertionClassifierAdapter,
    RuleRelationExtractorAdapter,
)
from medical_kg_nlp.governance import GovernancePolicy, fingerprint_artifact, safe_artifact_path, verify_artifact
from medical_kg_nlp.governance.audit import AuditEvent, InMemoryAuditSink
from medical_kg_nlp.context.assertion import AssertionClassifier
from medical_kg_nlp.kg.validator import KGValidator
from medical_kg_nlp.kg.ontology_reasoner import OntologyReasoner
from medical_kg_nlp.pipeline.components import PipelineComponents
from medical_kg_nlp.pipeline.builders import (
    build_entity_extractor,
    build_graph_repository,
    build_linking,
    build_terminology,
)
from medical_kg_nlp.pipeline.config_schema import validate_pipeline_mapping
from medical_kg_nlp.pipeline.model_config import PipelineModelConfig
from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.pipeline.runner import PipelineRunner
from medical_kg_nlp.pipeline.runtime import Closable, PipelineRuntime
from medical_kg_nlp.preprocessing.normalizer import (
    DEFAULT_NORMALIZATION_CONTRACT,
    NORMALIZATION_CONTRACT_VERSION,
    NormalizationContract,
)
from medical_kg_nlp.relations.rule_relations import RuleRelationExtractor

__all__ = ["PipelineFactory", "PipelineConfig", "TerminologyConfig"]


@dataclass(frozen=True)
class TerminologyConfig:
    """Paths and bounded lookup settings owned by the terminology subsystem."""

    recognition_dictionary_path: str = "data/dictionaries/seed_concepts.jsonl"
    normalization_dictionary_paths: tuple[str, ...] = ()
    normalization_index_path: str | None = None
    normalization_alias_overlay_paths: tuple[str, ...] = ()
    knowledge_graph_index_path: str | None = None
    terminology_cache_dir: str = ".cache/medical-kg/terminology"
    terminology_query_cache_size: int = 0
    # Benchmark-specific reviewed memory is terminal on match, so reusable
    # profiles must opt in with an explicit, versioned artifact.
    reviewed_mention_path: str | None = None
    mention_code_memory_path: str | None = None
    learned_edit_path: str | None = None
    synonym_index_path: str | None = None
    synonym_index_terminology_fingerprint: str | None = None
    additional_recognition_dictionary_path: str | None = None
    additional_recognition_dictionary_paths: tuple[str, ...] = ()
    abbreviation_path: str = "data/dictionaries/abbreviations.jsonl"
    alias_overlay_path: str | None = "data/dictionaries/vietnamese_medical_alias.jsonl"
    contextual_alias_path: str | None = None
    false_positive_path: str | None = None

    def __post_init__(self) -> None:
        if self.terminology_query_cache_size < 0:
            raise ValueError("terminology.query_cache_size must be non-negative")


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable composition configuration grouped by subsystem ownership."""

    terminology: TerminologyConfig = field(default_factory=TerminologyConfig)
    pipeline_version: str = "0.2.0"
    options: PipelineOptions = field(default_factory=PipelineOptions)
    models: PipelineModelConfig = field(default_factory=PipelineModelConfig)
    normalization_contract: NormalizationContract = DEFAULT_NORMALIZATION_CONTRACT
    governance: GovernancePolicy = GovernancePolicy()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PipelineConfig":
        validate_pipeline_mapping(payload)
        terminology = _mapping(payload.get("terminology"), "terminology")
        pipeline = _mapping(payload.get("pipeline"), "pipeline")
        models = _mapping(payload.get("models"), "models")
        normalization = _mapping(payload.get("normalization"), "normalization")
        governance_payload = payload.get("governance", {})
        if not isinstance(governance_payload, Mapping):
            raise ValueError("governance must be a mapping")
        normalization_version = _string(
            normalization,
            "version",
            DEFAULT_NORMALIZATION_CONTRACT.version,
        )
        if normalization_version != NORMALIZATION_CONTRACT_VERSION:
            raise ValueError(
                "Unsupported normalization.version: "
                f"{normalization_version!r}; expected {NORMALIZATION_CONTRACT_VERSION!r}"
            )
        return cls(
            terminology=TerminologyConfig(
                recognition_dictionary_path=_string(
                    terminology,
                    "recognition_path",
                    TerminologyConfig.recognition_dictionary_path,
                ),
                normalization_dictionary_paths=_string_tuple(
                    terminology.get("normalization_paths"), "normalization_paths"
                ),
                normalization_index_path=_optional_string(
                    terminology.get("normalization_index_path")
                ),
                normalization_alias_overlay_paths=_string_tuple(
                    terminology.get("normalization_alias_overlay_paths"),
                    "normalization_alias_overlay_paths",
                ),
                knowledge_graph_index_path=_optional_string(
                    terminology.get("knowledge_graph_index_path")
                ),
                terminology_cache_dir=_string(
                    terminology,
                    "cache_dir",
                    TerminologyConfig.terminology_cache_dir,
                ),
                terminology_query_cache_size=_nonnegative_int(
                    terminology,
                    "query_cache_size",
                    TerminologyConfig.terminology_query_cache_size,
                ),
                reviewed_mention_path=_optional_string(terminology.get("reviewed_mention_path")),
                mention_code_memory_path=_optional_string(terminology.get("mention_code_memory_path")),
                learned_edit_path=_optional_string(terminology.get("learned_edit_path")),
                synonym_index_path=_optional_string(terminology.get("synonym_index_path")),
                synonym_index_terminology_fingerprint=_optional_string(
                    terminology.get("synonym_index_terminology_fingerprint")
                ),
                additional_recognition_dictionary_path=_optional_string(
                    terminology.get("additional_recognition_path")
                ),
                additional_recognition_dictionary_paths=_string_tuple(
                    terminology.get("additional_recognition_paths"),
                    "additional_recognition_paths",
                ),
                abbreviation_path=_string(
                    terminology,
                    "abbreviation_path",
                    TerminologyConfig.abbreviation_path,
                ),
                alias_overlay_path=_optional_string(terminology.get("alias_overlay_path")),
                contextual_alias_path=_optional_string(terminology.get("contextual_alias_path")),
                false_positive_path=_optional_string(terminology.get("false_positive_path")),
            ),
            pipeline_version=_string(pipeline, "version", cls.pipeline_version),
            options=PipelineOptions.from_mapping(pipeline),
            models=PipelineModelConfig.from_mapping(models),
            normalization_contract=NormalizationContract(version=normalization_version),
            governance=GovernancePolicy.from_mapping(governance_payload),
        )


class PipelineFactory:
    """Build a runner from configuration without leaking IO into orchestration."""

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig | Mapping[str, object] | None = None,
    ) -> PipelineRunner:
        resolved = cls._resolve(config)
        _verify_configured_artifacts(resolved)
        audit_sink = InMemoryAuditSink()
        audit_sink.emit(
            AuditEvent(
                "profile_load",
                profile_fingerprint=_configuration_fingerprint(resolved),
            )
        )
        terminology = build_terminology(resolved)
        audit_sink.emit(
            AuditEvent("terminology_load", artifact_sha256=terminology.fingerprint)
        )
        options = resolved.options
        entity = build_entity_extractor(resolved, terminology)
        graph = build_graph_repository(resolved)
        linking = build_linking(resolved, terminology, graph)
        assertion_classifier = (
            RuleAssertionClassifierAdapter(AssertionClassifier())
            if options.enable_context
            else None
        )

        relation_extractor = (
            RuleRelationExtractorAdapter(RuleRelationExtractor())
            if options.enable_relations
            else None
        )
        knowledge_validator = (
            KGValidatorAdapter(
                KGValidator(
                    OntologyReasoner(terminology.recognition_store)
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
            entity_extractor=entity.extractor,
            assertion_classifier=assertion_classifier,
            candidate_retriever=linking.candidate_adapter,
            candidate_reranker=linking.candidate_reranker,
            document_candidate_reranker=graph.document_reranker,
            candidate_assigner=linking.candidate_adapter,
            relation_extractor=relation_extractor,
            knowledge_validator=knowledge_validator,
            terminology_repository=terminology.repository,
            options=options,
            normalization_contract=resolved.normalization_contract,
            pipeline_version=resolved.pipeline_version,
            configuration_fingerprint=_configuration_fingerprint(resolved),
            terminology_fingerprint=terminology.fingerprint,
            model_revision=_model_revisions(resolved),
            backend="local",
            audit_sink=audit_sink,
        )
        for model in _model_configs(resolved):
            model_fingerprint, fingerprint_kind = _model_fingerprint(model)
            audit_sink.emit(
                AuditEvent(
                    "model_load",
                    model_revision=model.revision,
                    details={
                        "model_id": model.model_id,
                        "fingerprint": model_fingerprint,
                        "fingerprint_kind": fingerprint_kind,
                    },
                )
            )
        resources = _unique_closables(
            (
                terminology.repository,
                graph.repository,
                entity.extractor,
                linking.dense_retriever,
                linking.candidate_reranker,
                graph.document_reranker,
            )
        )
        runner = PipelineRunner(components)
        runner.attach_resources(resources)
        return runner

    @classmethod
    def runtime_from_config(
        cls,
        config: PipelineConfig | Mapping[str, object] | None = None,
    ) -> PipelineRuntime:
        """Build an explicitly owned runtime for long-lived and CLI execution."""

        runner = cls.from_config(config)
        return PipelineRuntime(runner, runner.resources)

    @staticmethod
    def _resolve(
        config: PipelineConfig | Mapping[str, object] | None,
    ) -> PipelineConfig:
        if config is None:
            return PipelineConfig()
        if isinstance(config, PipelineConfig):
            return config
        return PipelineConfig.from_mapping(config)


def _mapping(value: object, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _configuration_fingerprint(config: PipelineConfig) -> str:
    payload = asdict(config)
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _model_revisions(config: PipelineConfig) -> str | None:
    revisions: list[str] = []
    for model in (
        config.models.entity_extractor,
        config.models.candidate_reranker,
        config.models.candidate_dense_encoder,
    ):
        if model is not None:
            revisions.append(model.revision)
    if config.models.candidate_listwise_reranker is not None:
        revisions.append(config.models.candidate_listwise_reranker.model.revision)
    return ",".join(revisions) if revisions else None


def _model_configs(config: PipelineConfig) -> tuple[Any, ...]:
    values = [
        model
        for model in (
            config.models.entity_extractor,
            config.models.candidate_reranker,
            config.models.candidate_dense_encoder,
        )
        if model is not None
    ]
    if config.models.candidate_listwise_reranker is not None:
        values.append(config.models.candidate_listwise_reranker.model)
    return tuple(values)


def _model_fingerprint(model: Any) -> tuple[str, str]:
    """Fingerprint local model bytes or the pinned remote identity, never model output."""

    path = Path(model.model_id).expanduser()
    if path.is_file() or path.is_dir():
        return fingerprint_artifact(path), "local_artifact"
    identity = f"{model.model_id}@{model.revision}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest(), "pinned_identity"


def _verify_configured_artifacts(config: PipelineConfig) -> None:
    """Apply explicit allowlist checks before any model or terminology load."""

    if not config.governance.artifact_allowlist:
        return
    if not config.governance.allowed_artifact_roots:
        raise ValueError("artifact_allowlist requires governance.allowed_artifact_roots")
    for raw_path, expected_sha256 in config.governance.artifact_allowlist:
        path = safe_artifact_path(
            raw_path,
            allowed_roots=config.governance.allowed_artifact_roots,
        )
        verify_artifact(path, expected_sha256)




def _unique_closables(values: tuple[object, ...]) -> tuple[Closable, ...]:
    """Collect closeable composition resources without imposing ownership on pure adapters."""

    output: list[Closable] = []
    seen: set[int] = set()
    for value in values:
        close = getattr(value, "close", None)
        if not callable(close) or id(value) in seen:
            continue
        seen.add(id(value))
        output.append(value)  # type: ignore[arg-type]
    return tuple(output)


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


def _nonnegative_int(payload: Mapping[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value
