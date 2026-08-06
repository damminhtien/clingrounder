"""Strict subsystem configuration records for the reusable pipeline.

The YAML boundary is intentionally grouped by ownership.  ``PipelineOptions`` remains the
compiled runtime representation consumed by the hot path; these records are the source-level
configuration contract and keep cross-feature policies next to the subsystem they control.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from medical_kg_nlp.pipeline.options import (
    DEFAULT_CANDIDATE_SOURCES,
    SUPPORTED_CANDIDATE_SOURCES,
    PipelineOptions,
)

__all__ = [
    "ContextConfig",
    "GraphEvidenceConfig",
    "LinkingConfig",
    "RelationsConfig",
    "RuntimeConfig",
    "ValidationConfig",
    "compile_pipeline_options",
    "parse_subsystem_configs",
]


Provider = Literal["disabled", "rules"]


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Assertion and local-context policy."""

    provider: Provider = "rules"
    context_window: int = 80
    lookup_normalization_diagnostics: bool = True

    @property
    def enabled(self) -> bool:
        return self.provider != "disabled"

    def __post_init__(self) -> None:
        if self.provider not in {"disabled", "rules"}:
            raise ValueError("context.provider must be 'disabled' or 'rules'")
        if self.context_window < 0:
            raise ValueError("context.context_window must be non-negative")


@dataclass(frozen=True, slots=True)
class LinkingConfig:
    """Candidate retrieval, reranking, and assignment policy."""

    provider: Provider = "rules"
    max_candidates: int = 20
    candidate_sources: tuple[str, ...] = DEFAULT_CANDIDATE_SOURCES
    assignment_threshold: float = 0.75
    assignment_margin: float = 0.05
    candidate_threshold: float = 0.75
    candidate_relative_margin: float = 0.05
    max_qualified_candidates: int = 5
    candidate_thresholds_by_type: tuple[tuple[str, float], ...] = ()
    candidate_thresholds_by_source: tuple[tuple[str, float], ...] = ()
    emit_probabilities_by_source: tuple[tuple[str, float], ...] = ()
    enforce_rxnorm_structure: bool = True
    candidate_reranker: Provider = "rules"

    @property
    def enabled(self) -> bool:
        return self.provider != "disabled"

    def __post_init__(self) -> None:
        if self.provider not in {"disabled", "rules"}:
            raise ValueError("linking.provider must be 'disabled' or 'rules'")
        if self.candidate_reranker not in {"disabled", "rules"}:
            raise ValueError("linking.reranker.provider must be 'disabled' or 'rules'")
        if not 1 <= self.max_candidates <= 1000:
            raise ValueError("linking.max_candidates must be between 1 and 1000")
        if not self.candidate_sources or len(set(self.candidate_sources)) != len(self.candidate_sources):
            raise ValueError("linking.candidate_sources must be non-empty and unique")
        unknown = set(self.candidate_sources) - SUPPORTED_CANDIDATE_SOURCES
        if unknown:
            raise ValueError(f"linking.candidate_sources contains unknown values: {sorted(unknown)}")
        if self.max_qualified_candidates < 1 or self.max_qualified_candidates > 5:
            raise ValueError("linking.max_qualified_candidates must be between 1 and 5")
        for name, value in (
            ("assignment_threshold", self.assignment_threshold),
            ("assignment_margin", self.assignment_margin),
            ("candidate_threshold", self.candidate_threshold),
            ("candidate_relative_margin", self.candidate_relative_margin),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"linking.{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class GraphEvidenceConfig:
    """Optional graph-backed candidate evidence policy."""

    provider: Provider = "disabled"
    max_bonus: float = 0.04
    min_support: int = 2
    relation_types: tuple[str, ...] = ("CO_OCCURS_WITH",)
    cache_size: int = 4096

    @property
    def enabled(self) -> bool:
        return self.provider != "disabled"

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_bonus <= 1.0:
            raise ValueError("graph.max_bonus must be between 0 and 1")
        if self.min_support < 1:
            raise ValueError("graph.min_support must be at least 1")
        if self.cache_size < 1:
            raise ValueError("graph.cache_size must be at least 1")
        if not self.relation_types or any(not item.strip() for item in self.relation_types):
            raise ValueError("graph.relation_types must contain non-empty values")


@dataclass(frozen=True, slots=True)
class RelationsConfig:
    """Relation extraction and relation-level KG validation policy."""

    provider: Provider = "rules"
    validate_with_kg: bool = True

    @property
    def enabled(self) -> bool:
        return self.provider != "disabled"


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Entity and relation validation policy."""

    validate_entities_with_kg: bool = True
    validate_relations_with_kg: bool = True


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Execution defaults kept separate from NLP and terminology policy."""

    backend: Literal["serial", "thread", "process"] = "serial"
    workers: int = 1
    chunksize: int = 4
    fail_fast: bool = True

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("runtime.workers must be at least 1")
        if self.chunksize < 1:
            raise ValueError("runtime.chunksize must be at least 1")


def parse_subsystem_configs(
    payload: Mapping[str, object],
) -> tuple[ContextConfig, LinkingConfig, GraphEvidenceConfig, RelationsConfig, ValidationConfig, RuntimeConfig]:
    """Parse grouped blocks while keeping old flat profile fields as a source mapping.

    This is a single parser, not a second runtime.  Existing benchmark-owned profiles can
    continue to provide flat fields until they are regenerated; reusable profiles should use
    the grouped blocks so ownership is visible to readers and inspectors.
    """

    context = _mapping(payload.get("context"), "pipeline.context")
    linking = _mapping(payload.get("linking"), "pipeline.linking")
    graph = _mapping(payload.get("graph"), "pipeline.graph")
    relations = _mapping(payload.get("relations"), "pipeline.relations")
    validation = _mapping(payload.get("validation"), "pipeline.validation")
    runtime = _mapping(payload.get("runtime"), "pipeline.runtime")
    reranker = _mapping(linking.get("reranker"), "pipeline.linking.reranker")

    flat = dict(payload)
    return (
        ContextConfig(
            provider=_provider(context, "provider", "rules" if _bool(flat, "enable_context", True) else "disabled"),
            context_window=_int(context, "context_window", _int(flat, "context_window", 80)),
            lookup_normalization_diagnostics=_bool(
                context,
                "lookup_normalization_diagnostics",
                _bool(flat, "enable_lookup_normalization_diagnostics", True),
            ),
        ),
        LinkingConfig(
            provider=_provider(linking, "provider", "rules" if _bool(flat, "enable_linking", True) else "disabled"),
            max_candidates=_int(linking, "max_candidates", _int(flat, "max_candidates", 20)),
            candidate_sources=_string_tuple(
                linking,
                "candidate_sources",
                _string_tuple(flat, "candidate_sources", DEFAULT_CANDIDATE_SOURCES),
            ),
            assignment_threshold=_float(linking, "assignment_threshold", _float(flat, "link_assignment_threshold", 0.75)),
            assignment_margin=_float(linking, "assignment_margin", _float(flat, "link_assignment_margin", 0.05)),
            candidate_threshold=_float(linking, "candidate_threshold", _float(flat, "link_candidate_threshold", 0.75)),
            candidate_relative_margin=_float(linking, "candidate_relative_margin", _float(flat, "link_candidate_relative_margin", 0.05)),
            max_qualified_candidates=_int(linking, "max_qualified_candidates", _int(flat, "link_max_qualified_candidates", 5)),
            candidate_thresholds_by_type=_thresholds(linking, "candidate_thresholds_by_type", _thresholds(flat, "link_candidate_thresholds_by_type", ())),
            candidate_thresholds_by_source=_thresholds(linking, "candidate_thresholds_by_source", _thresholds(flat, "link_candidate_thresholds_by_source", ())),
            emit_probabilities_by_source=_thresholds(linking, "emit_probabilities_by_source", _thresholds(flat, "link_emit_probabilities_by_source", ())),
            enforce_rxnorm_structure=_bool(linking, "enforce_rxnorm_structure", _bool(flat, "link_enforce_rxnorm_structure", True)),
            candidate_reranker=_provider(
                reranker if "reranker" in linking else linking,
                "provider",
                "rules" if _bool(flat, "enable_candidate_reranking", True) else "disabled",
            ),
        ),
        GraphEvidenceConfig(
            provider=_provider(graph, "provider", "rules" if _bool(flat, "enable_graph_evidence_reranking", False) else "disabled"),
            max_bonus=_float(graph, "max_bonus", _float(flat, "graph_evidence_max_bonus", 0.04)),
            min_support=_int(graph, "min_support", _int(flat, "graph_evidence_min_support", 2)),
            relation_types=_string_tuple(graph, "relation_types", _string_tuple(flat, "graph_evidence_relation_types", ("CO_OCCURS_WITH",))),
            cache_size=_int(graph, "cache_size", _int(flat, "graph_evidence_cache_size", 4096)),
        ),
        RelationsConfig(
            provider=_provider(relations, "provider", "rules" if _bool(flat, "enable_relations", True) else "disabled"),
            validate_with_kg=_bool(relations, "validate_with_kg", _bool(flat, "enable_relation_kg_validation", True)),
        ),
        ValidationConfig(
            validate_entities_with_kg=_bool(validation, "entities_with_kg", _bool(flat, "enable_entity_kg_validation", True)),
            validate_relations_with_kg=_bool(validation, "relations_with_kg", _bool(flat, "enable_relation_kg_validation", True)),
        ),
        RuntimeConfig(
            backend=cast(
                Literal["serial", "thread", "process"],
                str(runtime.get("backend", "serial")),
            ),
            workers=_int(runtime, "workers", 1),
            chunksize=_int(runtime, "chunksize", 4),
            fail_fast=_bool(runtime, "fail_fast", True),
        ),
    )


def compile_pipeline_options(
    context: ContextConfig,
    linking: LinkingConfig,
    graph: GraphEvidenceConfig,
    relations: RelationsConfig,
    validation: ValidationConfig,
) -> PipelineOptions:
    """Compile subsystem policies into the one runtime options record."""

    return PipelineOptions(
        max_candidates=linking.max_candidates,
        context_window=context.context_window,
        link_assignment_threshold=linking.assignment_threshold,
        link_assignment_margin=linking.assignment_margin,
        link_candidate_threshold=linking.candidate_threshold,
        link_candidate_relative_margin=linking.candidate_relative_margin,
        link_max_qualified_candidates=linking.max_qualified_candidates,
        link_candidate_thresholds_by_type=linking.candidate_thresholds_by_type,
        link_candidate_thresholds_by_source=linking.candidate_thresholds_by_source,
        link_emit_probabilities_by_source=linking.emit_probabilities_by_source,
        link_enforce_rxnorm_structure=linking.enforce_rxnorm_structure,
        candidate_sources=linking.candidate_sources,
        enable_lookup_normalization_diagnostics=context.lookup_normalization_diagnostics,
        enable_context=context.enabled,
        enable_linking=linking.enabled,
        enable_candidate_reranking=linking.candidate_reranker != "disabled",
        enable_graph_evidence_reranking=graph.enabled,
        graph_evidence_max_bonus=graph.max_bonus,
        graph_evidence_min_support=graph.min_support,
        graph_evidence_relation_types=graph.relation_types,
        graph_evidence_cache_size=graph.cache_size,
        enable_entity_kg_validation=validation.validate_entities_with_kg,
        enable_relations=relations.enabled,
        enable_relation_kg_validation=relations.validate_with_kg,
    )


def _mapping(value: object, path: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _provider(payload: Mapping[str, object], key: str, default: str) -> Provider:
    value = payload.get(key, default)
    if value not in {"disabled", "rules"}:
        raise ValueError(f"provider must be 'disabled' or 'rules', got {value!r}")
    return value  # type: ignore[return-value]


def _bool(payload: Mapping[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _int(payload: Mapping[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _float(payload: Mapping[str, object], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _string_tuple(payload: Mapping[str, object], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = payload.get(key, default)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{key} must be an array of non-empty strings")
    return tuple(value)


def _thresholds(payload: Mapping[str, object], key: str, default: tuple[tuple[str, float], ...]) -> tuple[tuple[str, float], ...]:
    value = payload.get(key, default)
    if isinstance(value, Mapping):
        return tuple(sorted((str(name), _float({"value": threshold}, "value", 0.0)) for name, threshold in value.items()))
    if isinstance(value, tuple):
        return value
    raise ValueError(f"{key} must be a mapping")
