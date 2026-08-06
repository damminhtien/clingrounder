"""Path-stable loading and rebasing for reusable pipeline YAML profiles."""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from medical_kg_nlp.pipeline.factory import PipelineConfig
from medical_kg_nlp.pipeline.config_schema import validate_pipeline_mapping
from medical_kg_nlp.pipeline.profile import (
    PIPELINE_PROFILE_SCHEMA_VERSION,
    PipelineProfileMetadata,
)
from medical_kg_nlp.utils.io import read_yaml

__all__ = ["ResolvedPipelineConfig"]

_TERMINOLOGY_SINGLE_PATHS = (
    "recognition_path",
    "normalization_index_path",
    "knowledge_graph_index_path",
    "cache_dir",
    "reviewed_mention_path",
    "mention_code_memory_path",
    "learned_edit_path",
    "synonym_index_path",
    "additional_recognition_path",
    "abbreviation_path",
    "alias_overlay_path",
    "contextual_alias_path",
    "false_positive_path",
)
_TERMINOLOGY_PATH_LISTS = (
    "normalization_paths",
    "normalization_alias_overlay_paths",
    "additional_recognition_paths",
)
_MODEL_BLOCKS = (
    "entity_extractor",
    "candidate_reranker",
    "candidate_listwise_reranker",
    "candidate_dense_encoder",
)


@dataclass(frozen=True)
class ResolvedPipelineConfig:
    """A pipeline profile whose filesystem references no longer depend on ``cwd``."""

    source_path: Path
    payload: dict[str, Any]
    factory_config: PipelineConfig
    profile: PipelineProfileMetadata | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        require_profile: bool = False,
    ) -> "ResolvedPipelineConfig":
        """Load one YAML profile and resolve every declared path from its parent.

        Reusable CLI profiles require metadata. Historical benchmark loaders may
        opt out while their configs are migrated into benchmark-owned storage.
        """

        source_path = Path(path).expanduser().resolve()
        raw = read_yaml(source_path)
        if not isinstance(raw, Mapping):
            raise ValueError("Pipeline config must be a YAML mapping")
        profile_payload = raw.get("profile")
        if require_profile and profile_payload is None:
            raise ValueError("Reusable pipeline config requires a profile block")
        if require_profile or profile_payload is not None:
            _validate_top_level_keys(raw)
            validate_pipeline_mapping(raw, require_schema_version=require_profile)
        if profile_payload is None:
            profile = None
        else:
            profile = PipelineProfileMetadata.from_mapping(
                _required_mapping(profile_payload, "profile")
            )
        payload = _map_declared_paths(
            raw,
            lambda value: _resolve_path(source_path.parent, value),
        )
        parsed = PipelineConfig.from_mapping(payload)
        return cls(
            source_path=source_path,
            payload=payload,
            factory_config=_resolve_factory_defaults(parsed, source_path.parent),
            profile=profile,
        )

    def inspection_report(self) -> dict[str, Any]:
        """Describe effective settings and filesystem dependencies without running NLP."""

        config = self.factory_config
        terminology = config.terminology
        return {
            "schema_version": PIPELINE_PROFILE_SCHEMA_VERSION,
            "profile_sha256": _file_sha256(self.source_path),
            "source": {
                "path": str(self.source_path),
                "sha256": _file_sha256(self.source_path),
            },
            "profile": None if self.profile is None else self.profile.to_dict(),
            "resources": _resource_report(config),
            "effective_config": _json_ready(
                {
                    "terminology": {
                        "recognition_path": terminology.recognition_dictionary_path,
                        "normalization_paths": terminology.normalization_dictionary_paths,
                        "normalization_index_path": terminology.normalization_index_path,
                        "normalization_alias_overlay_paths": (
                            terminology.normalization_alias_overlay_paths
                        ),
                        "knowledge_graph_index_path": terminology.knowledge_graph_index_path,
                        "cache_dir": terminology.terminology_cache_dir,
                        "query_cache_size": terminology.terminology_query_cache_size,
                        "reviewed_mention_path": terminology.reviewed_mention_path,
                        "mention_code_memory_path": terminology.mention_code_memory_path,
                        "learned_edit_path": terminology.learned_edit_path,
                        "synonym_index_path": terminology.synonym_index_path,
                        "synonym_index_terminology_fingerprint": (
                            terminology.synonym_index_terminology_fingerprint
                        ),
                        "additional_recognition_path": (
                            terminology.additional_recognition_dictionary_path
                        ),
                        "additional_recognition_paths": (
                            terminology.additional_recognition_dictionary_paths
                        ),
                        "abbreviation_path": terminology.abbreviation_path,
                        "alias_overlay_path": terminology.alias_overlay_path,
                        "contextual_alias_path": terminology.contextual_alias_path,
                        "false_positive_path": terminology.false_positive_path,
                    },
                    "pipeline": {
                        "version": config.pipeline_version,
                        **asdict(config.options),
                    },
                    "models": asdict(config.models),
                    "normalization_contract": asdict(config.normalization_contract),
                    "governance": asdict(config.governance),
                }
            ),
            "origins": _origin_report(self.payload),
        }

    def payload_for(self, destination: str | Path) -> dict[str, Any]:
        """Rebase explicit path fields for a derived YAML artifact.

        Relative references keep a calibrated artifact portable when the repository
        tree is moved as a unit. Remote Hugging Face identifiers remain unchanged.
        """

        destination_parent = Path(destination).expanduser().resolve().parent
        materialized = _materialize_terminology_paths(
            self.payload,
            self.factory_config,
        )
        return _map_declared_paths(
            materialized,
            lambda value: _relative_path(destination_parent, value),
        )


def _materialize_terminology_paths(
    payload: Mapping[str, object],
    config: PipelineConfig,
) -> dict[str, Any]:
    """Persist effective defaults before a profile is moved to another directory."""

    output = copy.deepcopy(dict(payload))
    terminology = _optional_mapping(output.get("terminology"), "terminology")
    terminology_config = config.terminology
    terminology.update(
        {
            "recognition_path": terminology_config.recognition_dictionary_path,
            "normalization_paths": list(terminology_config.normalization_dictionary_paths),
            "normalization_index_path": terminology_config.normalization_index_path,
            "normalization_alias_overlay_paths": list(
                terminology_config.normalization_alias_overlay_paths
            ),
            "knowledge_graph_index_path": terminology_config.knowledge_graph_index_path,
            "cache_dir": terminology_config.terminology_cache_dir,
            "reviewed_mention_path": terminology_config.reviewed_mention_path,
            "mention_code_memory_path": terminology_config.mention_code_memory_path,
            "learned_edit_path": terminology_config.learned_edit_path,
            "synonym_index_path": terminology_config.synonym_index_path,
            "synonym_index_terminology_fingerprint": (
                terminology_config.synonym_index_terminology_fingerprint
            ),
            "additional_recognition_path": (
                terminology_config.additional_recognition_dictionary_path
            ),
            "additional_recognition_paths": list(
                terminology_config.additional_recognition_dictionary_paths
            ),
            "abbreviation_path": terminology_config.abbreviation_path,
            "alias_overlay_path": terminology_config.alias_overlay_path,
            "contextual_alias_path": terminology_config.contextual_alias_path,
            "false_positive_path": terminology_config.false_positive_path,
        }
    )
    output["terminology"] = terminology
    return output


def _map_declared_paths(
    payload: Mapping[str, object],
    transform: Callable[[str], str],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(payload))
    terminology = _optional_mapping(output.get("terminology"), "terminology")
    for key in _TERMINOLOGY_SINGLE_PATHS:
        value = terminology.get(key)
        if value is not None:
            terminology[key] = transform(_path_string(value, f"terminology.{key}"))
    for key in _TERMINOLOGY_PATH_LISTS:
        value = terminology.get(key)
        if value is None:
            continue
        if not isinstance(value, list | tuple):
            raise ValueError(f"terminology.{key} must be an array")
        terminology[key] = [
            transform(_path_string(item, f"terminology.{key}")) for item in value
        ]
    if terminology or "terminology" in output:
        output["terminology"] = terminology

    models = _optional_mapping(output.get("models"), "models")
    for block_name in _MODEL_BLOCKS:
        raw_block = models.get(block_name)
        if raw_block is None:
            continue
        block = _required_mapping(raw_block, f"models.{block_name}")
        run_spec = block.get("run_spec")
        if run_spec is not None:
            block["run_spec"] = transform(
                _path_string(run_spec, f"models.{block_name}.run_spec")
            )
        model_id = block.get("model_id")
        if model_id is not None:
            raw_model_id = _path_string(model_id, f"models.{block_name}.model_id")
            # MODEL: a run specification verifies a local training artifact, while a
            # bare ``organization/model`` identifier belongs to a remote model registry.
            if run_spec is not None or _is_explicit_local_model_reference(raw_model_id):
                block["model_id"] = transform(raw_model_id)
        models[block_name] = block
    if models or "models" in output:
        output["models"] = models
    governance = _optional_mapping(output.get("governance"), "governance")
    roots = governance.get("allowed_artifact_roots")
    if roots is not None:
        if not isinstance(roots, list | tuple):
            raise ValueError("governance.allowed_artifact_roots must be an array")
        governance["allowed_artifact_roots"] = [
            transform(_path_string(item, "governance.allowed_artifact_roots"))
            for item in roots
        ]
    allowlist = governance.get("artifact_allowlist")
    if allowlist is not None:
        if not isinstance(allowlist, list | tuple):
            raise ValueError("governance.artifact_allowlist must be an array")
        governance["artifact_allowlist"] = [
            {
                **_required_mapping(item, "governance.artifact_allowlist[]"),
                "path": transform(
                    _path_string(
                        _required_mapping(item, "governance.artifact_allowlist[]")["path"],
                        "governance.artifact_allowlist[].path",
                    )
                ),
            }
            for item in allowlist
        ]
    if governance or "governance" in output:
        output["governance"] = governance
    return output


def _resolve_factory_defaults(
    config: PipelineConfig,
    base_dir: Path,
) -> PipelineConfig:
    """Resolve dataclass defaults as well as paths explicitly present in YAML."""

    terminology = config.terminology
    resolved_terminology = replace(
        terminology,
        recognition_dictionary_path=_resolve_path(base_dir, terminology.recognition_dictionary_path),
        normalization_dictionary_paths=tuple(
            _resolve_path(base_dir, value)
            for value in terminology.normalization_dictionary_paths
        ),
        normalization_index_path=_resolve_optional(
            base_dir, terminology.normalization_index_path
        ),
        normalization_alias_overlay_paths=tuple(
            _resolve_path(base_dir, value)
            for value in terminology.normalization_alias_overlay_paths
        ),
        knowledge_graph_index_path=_resolve_optional(
            base_dir, terminology.knowledge_graph_index_path
        ),
        terminology_cache_dir=_resolve_path(base_dir, terminology.terminology_cache_dir),
        reviewed_mention_path=_resolve_optional(base_dir, terminology.reviewed_mention_path),
        mention_code_memory_path=_resolve_optional(
            base_dir, terminology.mention_code_memory_path
        ),
        learned_edit_path=_resolve_optional(base_dir, terminology.learned_edit_path),
        synonym_index_path=_resolve_optional(base_dir, terminology.synonym_index_path),
        additional_recognition_dictionary_path=_resolve_optional(
            base_dir, terminology.additional_recognition_dictionary_path
        ),
        additional_recognition_dictionary_paths=tuple(
            _resolve_path(base_dir, value)
            for value in terminology.additional_recognition_dictionary_paths
        ),
        abbreviation_path=_resolve_path(base_dir, terminology.abbreviation_path),
        alias_overlay_path=_resolve_optional(base_dir, terminology.alias_overlay_path),
    )
    resolved_governance = replace(
        config.governance,
        allowed_artifact_roots=tuple(
            _resolve_path(base_dir, value)
            for value in config.governance.allowed_artifact_roots
        ),
        artifact_allowlist=tuple(
            (_resolve_path(base_dir, path), digest)
            for path, digest in config.governance.artifact_allowlist
        ),
    )
    return replace(
        config,
        terminology=resolved_terminology,
        governance=resolved_governance,
    )


def _resolve_optional(base_dir: Path, value: str | None) -> str | None:
    return None if value is None else _resolve_path(base_dir, value)


def _resolve_path(base_dir: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return str(candidate.resolve())


def _relative_path(base_dir: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        return value
    return os.path.relpath(candidate, start=base_dir)


def _is_explicit_local_model_reference(value: str) -> bool:
    candidate = Path(value).expanduser()
    return (
        candidate.is_absolute()
        or value.startswith(("./", "../", "~"))
    )


def _path_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty path string")
    return value


def _optional_mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _required_mapping(value, name)


def _required_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _validate_top_level_keys(payload: Mapping[str, object]) -> None:
    allowed = {
        "schema_version",
        "profile",
        "terminology",
        "pipeline",
        "models",
        "normalization",
        "governance",
        # Historical benchmark profiles retain immutable experiment metadata.
        "provenance",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown pipeline config keys: {', '.join(unknown)}")


def _resource_report(config: PipelineConfig) -> list[dict[str, object]]:
    values: list[tuple[str, str | None]] = [
        ("terminology.recognition_path", config.terminology.recognition_dictionary_path),
        ("terminology.normalization_index_path", config.terminology.normalization_index_path),
        ("terminology.knowledge_graph_index_path", config.terminology.knowledge_graph_index_path),
        ("terminology.reviewed_mention_path", config.terminology.reviewed_mention_path),
        ("terminology.mention_code_memory_path", config.terminology.mention_code_memory_path),
        ("terminology.learned_edit_path", config.terminology.learned_edit_path),
        ("terminology.synonym_index_path", config.terminology.synonym_index_path),
        (
            "terminology.additional_recognition_path",
            config.terminology.additional_recognition_dictionary_path,
        ),
        ("terminology.abbreviation_path", config.terminology.abbreviation_path),
        ("terminology.alias_overlay_path", config.terminology.alias_overlay_path),
        ("terminology.contextual_alias_path", config.terminology.contextual_alias_path),
        ("terminology.false_positive_path", config.terminology.false_positive_path),
    ]
    values.extend(
        (f"terminology.normalization_paths[{index}]", value)
        for index, value in enumerate(config.terminology.normalization_dictionary_paths)
    )
    for field_name, model in (
        ("models.entity_extractor", config.models.entity_extractor),
        ("models.candidate_reranker", config.models.candidate_reranker),
        ("models.candidate_dense_encoder", config.models.candidate_dense_encoder),
    ):
        if model is not None and _is_explicit_local_model_reference(model.model_id):
            values.append((f"{field_name}.model_id", model.model_id))
    values.extend(
        (f"terminology.normalization_alias_overlay_paths[{index}]", value)
        for index, value in enumerate(config.terminology.normalization_alias_overlay_paths)
    )
    values.extend(
        (f"terminology.additional_recognition_paths[{index}]", value)
        for index, value in enumerate(config.terminology.additional_recognition_dictionary_paths)
    )
    report: list[dict[str, object]] = []
    for field_name, raw_path in values:
        if raw_path is None:
            continue
        path = Path(raw_path)
        report.append(
            {
                "field": field_name,
                "path": str(path),
                "exists": path.exists(),
                "kind": "directory" if path.is_dir() else "file",
                "sha256": _resource_sha256(path),
            }
        )
    return report


def _json_ready(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_sha256(path: Path) -> str | None:
    """Fingerprint files without making directory scans part of profile loading."""

    if not path.is_file():
        return None
    return _file_sha256(path)


def _origin_report(payload: Mapping[str, object]) -> dict[str, dict[str, str]]:
    """Expose whether inspectable settings came from YAML or a typed default."""

    sections = {
        "terminology": tuple(_TERMINOLOGY_SINGLE_PATHS) + tuple(_TERMINOLOGY_PATH_LISTS),
        "pipeline": (
            "version",
            "max_candidates",
            "context_window",
            "candidate_sources",
            "enable_context",
            "enable_linking",
            "enable_candidate_reranking",
            "enable_graph_evidence_reranking",
            "enable_entity_kg_validation",
            "enable_relations",
            "enable_relation_kg_validation",
        ),
        "models": tuple(_MODEL_BLOCKS),
        "normalization": ("version",),
    }
    result: dict[str, dict[str, str]] = {}
    for section, keys in sections.items():
        raw = payload.get(section)
        mapping = raw if isinstance(raw, Mapping) else {}
        result[section] = {
            key: "explicit" if key in mapping else "default" for key in keys
        }
    return result
