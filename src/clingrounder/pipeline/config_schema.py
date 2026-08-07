"""Strict structural checks shared by reusable pipeline configuration loaders.

The runtime still exposes frozen dataclass configuration objects.  This module is the
schema boundary: it rejects unknown keys before defaults are applied, so a typo cannot
silently change a production profile.
"""

from __future__ import annotations

from collections.abc import Mapping

from clingrounder.pipeline.profile import PIPELINE_PROFILE_SCHEMA_VERSION

__all__ = ["validate_pipeline_mapping"]

_TERMINOLOGY_KEYS = {
    "recognition_path",
    "normalization_paths",
    "normalization_index_path",
    "normalization_alias_overlay_paths",
    "knowledge_graph_index_path",
    "cache_dir",
    "query_cache_size",
    "reviewed_mention_path",
    "mention_code_memory_path",
    "learned_edit_path",
    "synonym_index_path",
    "synonym_index_terminology_fingerprint",
    "additional_recognition_path",
    "additional_recognition_paths",
    "abbreviation_path",
    "alias_overlay_path",
    "contextual_alias_path",
    "false_positive_path",
}

_GOVERNANCE_KEYS = {
    "data",
    "allowed_artifact_roots",
    "artifact_allowlist",
    "local_files_only",
}

_PIPELINE_KEYS = {
    "version",
    "context",
    "linking",
    "graph",
    "relations",
    "validation",
    "runtime",
    "max_candidates",
    "context_window",
    "link_assignment_threshold",
    "link_assignment_margin",
    "link_candidate_threshold",
    "link_candidate_relative_margin",
    "link_max_qualified_candidates",
    "link_candidate_thresholds_by_type",
    "link_candidate_thresholds_by_source",
    "link_emit_probabilities_by_source",
    "link_enforce_rxnorm_structure",
    "candidate_sources",
    "enable_lookup_normalization_diagnostics",
    "enable_context",
    "enable_linking",
    "enable_candidate_reranking",
    "enable_graph_evidence_reranking",
    "graph_evidence_max_bonus",
    "graph_evidence_min_support",
    "graph_evidence_relation_types",
    "graph_evidence_cache_size",
    "enable_entity_kg_validation",
    "enable_relations",
    "enable_relation_kg_validation",
}

_CONTEXT_KEYS = {
    "provider",
    "context_window",
    "lookup_normalization_diagnostics",
}
_LINKING_KEYS = {
    "provider",
    "max_candidates",
    "candidate_sources",
    "assignment_threshold",
    "assignment_margin",
    "candidate_threshold",
    "candidate_relative_margin",
    "max_qualified_candidates",
    "candidate_thresholds_by_type",
    "candidate_thresholds_by_source",
    "emit_probabilities_by_source",
    "enforce_rxnorm_structure",
    "reranker",
}
_GRAPH_KEYS = {"provider", "max_bonus", "min_support", "relation_types", "cache_size"}
_RELATIONS_KEYS = {"provider", "validate_with_kg"}
_VALIDATION_KEYS = {"entities_with_kg", "relations_with_kg"}
_RUNTIME_KEYS = {"backend", "workers", "chunksize", "fail_fast"}

_MODEL_BLOCK_KEYS = {
    "model_id",
    "revision",
    "device",
    "batch_size",
    "max_length",
    "max_pairs_per_batch",
    "max_tokens",
    "subfolder",
    "run_spec",
    "stride",
    "default_confidence_threshold",
    "confidence_thresholds",
    "label_map",
    "combine_with_dictionary",
    "model_weight",
    "positive_label_index",
    "dtype",
    "local_files_only",
    "candidate_limit",
    "shuffle_seed",
    "structured_retries",
    "max_new_tokens",
    "temperature",
    "top_p",
    "seed",
    "enable_thinking",
}

_MODEL_KEYS = {
    "entity_extractor",
    "candidate_reranker",
    "candidate_listwise_reranker",
    "candidate_dense_encoder",
}

# Benchmark plugins own these fields.  They are intentionally accepted as metadata
# at the composition boundary but are never interpreted as reusable pipeline options.
_BENCHMARK_KEYS = {
    "input_dir",
    "output_dir",
    "zip",
    "run_root",
    "run_label",
    "mode",
    "dictionary",
    "abbreviations",
    "parallel",
    "runtime_metrics",
    "traces",
    "internal_predictions",
    "assertion_policy",
    "candidate_policy",
    "expected_count",
    "strict_validation",
    "max_candidates",
}


def validate_pipeline_mapping(
    payload: Mapping[str, object],
    *,
    require_schema_version: bool = False,
) -> None:
    """Reject unknown nested keys and, for profiles, require the current schema version."""

    if require_schema_version and payload.get("schema_version") != PIPELINE_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported pipeline profile schema_version: "
            f"{payload.get('schema_version')!r}; expected {PIPELINE_PROFILE_SCHEMA_VERSION!r}"
        )

    _check_keys(
        payload,
        {
            "schema_version",
            "profile",
            "terminology",
            "pipeline",
            "models",
            "normalization",
            "provenance",
            "governance",
            *_BENCHMARK_KEYS,
        },
        "",
    )
    for name, allowed in (
        ("terminology", _TERMINOLOGY_KEYS),
        ("pipeline", _PIPELINE_KEYS),
        ("governance", _GOVERNANCE_KEYS),
    ):
        value = payload.get(name)
        if value is not None:
            mapping = _check_mapping(value, name)
            _check_keys(mapping, allowed, name)
    pipeline = payload.get("pipeline")
    if pipeline is not None:
        pipeline_mapping = _check_mapping(pipeline, "pipeline")
        for name, allowed in (
            ("context", _CONTEXT_KEYS),
            ("linking", _LINKING_KEYS),
            ("graph", _GRAPH_KEYS),
            ("relations", _RELATIONS_KEYS),
            ("validation", _VALIDATION_KEYS),
            ("runtime", _RUNTIME_KEYS),
        ):
            block = pipeline_mapping.get(name)
            if block is not None:
                block_mapping = _check_mapping(block, f"pipeline.{name}")
                _check_keys(block_mapping, allowed, f"pipeline.{name}")
        linking = pipeline_mapping.get("linking")
        if linking is not None:
            linking_mapping = _check_mapping(linking, "pipeline.linking")
            reranker = linking_mapping.get("reranker")
            if reranker is not None:
                reranker_mapping = _check_mapping(reranker, "pipeline.linking.reranker")
                _check_keys(reranker_mapping, {"provider"}, "pipeline.linking.reranker")
    governance = payload.get("governance")
    if governance is not None:
        governance_mapping = _check_mapping(governance, "governance")
        data = governance_mapping.get("data")
        if data is not None:
            data_mapping = _check_mapping(data, "governance.data")
            _check_keys(
                data_mapping,
                {
                    "logging_level",
                    "text_retention",
                    "trace_retention",
                    "hash_document_ids",
                    "metadata_allowlist",
                    "deletion_behavior",
                },
                "governance.data",
            )

    models = payload.get("models")
    if models is not None:
        model_mapping = _check_mapping(models, "models")
        _check_keys(model_mapping, _MODEL_KEYS, "models")
        for name, block in model_mapping.items():
            if block is None:
                continue
            model_block = _check_mapping(block, f"models.{name}")
            _check_keys(model_block, _MODEL_BLOCK_KEYS, f"models.{name}")

    normalization = payload.get("normalization")
    if normalization is not None:
        normalization_mapping = _check_mapping(normalization, "normalization")
        _check_keys(normalization_mapping, {"version"}, "normalization")


def _check_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _check_keys(value: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        qualified = ", ".join(
            f"{path}.{key}" if path else key
            for key in unknown
        )
        raise ValueError(f"Unknown pipeline config keys: {qualified}")
