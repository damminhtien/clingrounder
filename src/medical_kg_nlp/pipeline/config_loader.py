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

from medical_kg_nlp.pipeline.factory import PipelineFactoryConfig
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
    "additional_recognition_path",
    "abbreviation_path",
    "alias_overlay_path",
    "contextual_alias_path",
)
_TERMINOLOGY_PATH_LISTS = (
    "normalization_paths",
    "normalization_alias_overlay_paths",
    "additional_recognition_paths",
)
_MODEL_BLOCKS = ("entity_extractor", "candidate_reranker")


@dataclass(frozen=True)
class ResolvedPipelineConfig:
    """A pipeline profile whose filesystem references no longer depend on ``cwd``."""

    source_path: Path
    payload: dict[str, Any]
    factory_config: PipelineFactoryConfig
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
        schema_version = raw.get("schema_version")
        profile_payload = raw.get("profile")
        if require_profile or profile_payload is not None:
            _validate_top_level_keys(raw)
        if profile_payload is None:
            if require_profile:
                raise ValueError("Reusable pipeline config requires a profile block")
            profile = None
        else:
            if schema_version != PIPELINE_PROFILE_SCHEMA_VERSION:
                raise ValueError(
                    "Pipeline profile schema_version must be "
                    f"{PIPELINE_PROFILE_SCHEMA_VERSION!r}"
                )
            profile = PipelineProfileMetadata.from_mapping(
                _required_mapping(profile_payload, "profile")
            )
        payload = _map_declared_paths(
            raw,
            lambda value: _resolve_path(source_path.parent, value),
        )
        parsed = PipelineFactoryConfig.from_mapping(payload)
        return cls(
            source_path=source_path,
            payload=payload,
            factory_config=_resolve_factory_defaults(parsed, source_path.parent),
            profile=profile,
        )

    def inspection_report(self) -> dict[str, Any]:
        """Describe effective settings and filesystem dependencies without running NLP."""

        config = self.factory_config
        return {
            "schema_version": PIPELINE_PROFILE_SCHEMA_VERSION,
            "source": {
                "path": str(self.source_path),
                "sha256": _file_sha256(self.source_path),
            },
            "profile": None if self.profile is None else self.profile.to_dict(),
            "resources": _resource_report(config),
            "effective_config": _json_ready(
                {
                    "terminology": {
                        "recognition_path": config.recognition_dictionary_path,
                        "normalization_paths": config.normalization_dictionary_paths,
                        "normalization_index_path": config.normalization_index_path,
                        "normalization_alias_overlay_paths": (
                            config.normalization_alias_overlay_paths
                        ),
                        "knowledge_graph_index_path": config.knowledge_graph_index_path,
                        "cache_dir": config.terminology_cache_dir,
                        "query_cache_size": config.terminology_query_cache_size,
                        "reviewed_mention_path": config.reviewed_mention_path,
                        "additional_recognition_path": (
                            config.additional_recognition_dictionary_path
                        ),
                        "additional_recognition_paths": (
                            config.additional_recognition_dictionary_paths
                        ),
                        "abbreviation_path": config.abbreviation_path,
                        "alias_overlay_path": config.alias_overlay_path,
                        "contextual_alias_path": config.contextual_alias_path,
                    },
                    "pipeline": {
                        "version": config.pipeline_version,
                        **asdict(config.options),
                    },
                    "models": asdict(config.models),
                    "normalization_contract": asdict(config.normalization_contract),
                }
            ),
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
    config: PipelineFactoryConfig,
) -> dict[str, Any]:
    """Persist effective defaults before a profile is moved to another directory."""

    output = copy.deepcopy(dict(payload))
    terminology = _optional_mapping(output.get("terminology"), "terminology")
    terminology.update(
        {
            "recognition_path": config.recognition_dictionary_path,
            "normalization_paths": list(config.normalization_dictionary_paths),
            "normalization_index_path": config.normalization_index_path,
            "normalization_alias_overlay_paths": list(
                config.normalization_alias_overlay_paths
            ),
            "knowledge_graph_index_path": config.knowledge_graph_index_path,
            "cache_dir": config.terminology_cache_dir,
            "reviewed_mention_path": config.reviewed_mention_path,
            "additional_recognition_path": (
                config.additional_recognition_dictionary_path
            ),
            "additional_recognition_paths": list(
                config.additional_recognition_dictionary_paths
            ),
            "abbreviation_path": config.abbreviation_path,
            "alias_overlay_path": config.alias_overlay_path,
            "contextual_alias_path": config.contextual_alias_path,
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
    return output


def _resolve_factory_defaults(
    config: PipelineFactoryConfig,
    base_dir: Path,
) -> PipelineFactoryConfig:
    """Resolve dataclass defaults as well as paths explicitly present in YAML."""

    return replace(
        config,
        recognition_dictionary_path=_resolve_path(
            base_dir, config.recognition_dictionary_path
        ),
        normalization_dictionary_paths=tuple(
            _resolve_path(base_dir, value)
            for value in config.normalization_dictionary_paths
        ),
        normalization_index_path=_resolve_optional(
            base_dir, config.normalization_index_path
        ),
        normalization_alias_overlay_paths=tuple(
            _resolve_path(base_dir, value)
            for value in config.normalization_alias_overlay_paths
        ),
        knowledge_graph_index_path=_resolve_optional(
            base_dir, config.knowledge_graph_index_path
        ),
        terminology_cache_dir=_resolve_path(base_dir, config.terminology_cache_dir),
        reviewed_mention_path=_resolve_optional(base_dir, config.reviewed_mention_path),
        additional_recognition_dictionary_path=_resolve_optional(
            base_dir, config.additional_recognition_dictionary_path
        ),
        additional_recognition_dictionary_paths=tuple(
            _resolve_path(base_dir, value)
            for value in config.additional_recognition_dictionary_paths
        ),
        abbreviation_path=_resolve_path(base_dir, config.abbreviation_path),
        alias_overlay_path=_resolve_optional(base_dir, config.alias_overlay_path),
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
        # Historical benchmark profiles retain immutable experiment metadata.
        "provenance",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown pipeline config keys: {', '.join(unknown)}")


def _resource_report(config: PipelineFactoryConfig) -> list[dict[str, object]]:
    values: list[tuple[str, str | None]] = [
        ("terminology.recognition_path", config.recognition_dictionary_path),
        ("terminology.normalization_index_path", config.normalization_index_path),
        ("terminology.knowledge_graph_index_path", config.knowledge_graph_index_path),
        ("terminology.reviewed_mention_path", config.reviewed_mention_path),
        (
            "terminology.additional_recognition_path",
            config.additional_recognition_dictionary_path,
        ),
        ("terminology.abbreviation_path", config.abbreviation_path),
        ("terminology.alias_overlay_path", config.alias_overlay_path),
        ("terminology.contextual_alias_path", config.contextual_alias_path),
    ]
    values.extend(
        (f"terminology.normalization_paths[{index}]", value)
        for index, value in enumerate(config.normalization_dictionary_paths)
    )
    values.extend(
        (f"terminology.normalization_alias_overlay_paths[{index}]", value)
        for index, value in enumerate(config.normalization_alias_overlay_paths)
    )
    values.extend(
        (f"terminology.additional_recognition_paths[{index}]", value)
        for index, value in enumerate(config.additional_recognition_dictionary_paths)
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
