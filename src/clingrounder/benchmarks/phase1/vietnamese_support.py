"""Build Qwen-only support evidence from a pinned Vietnamese source-task model."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.adapters import (
    HuggingFaceModelConfig,
    HuggingFaceSourceTokenClassifierAdapter,
)
from clingrounder.adapters.model_spans import ProjectedSourceEntity
from clingrounder.adapters.generative import ModelBudgetEntry
from clingrounder.benchmarks.phase1.qwen_proposals import Phase1Label
from clingrounder.benchmarks.phase1.round2 import (
    load_phase1_round2_documents,
)
from clingrounder.mining.io import load_documents, write_json, write_jsonl
from clingrounder.utils.hashing import sha256_file
from clingrounder.utils.io import read_yaml

__all__ = [
    "Phase1VietnameseSupportSpec",
    "build_phase1_vietnamese_model_support",
    "load_phase1_vietnamese_support_spec",
    "project_vietnamese_support_rows",
]

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_ALLOWED_PHASE1_LABELS = frozenset(
    {
        "TRIỆU_CHỨNG",
        "TÊN_XÉT_NGHIỆM",
        "KẾT_QUẢ_XÉT_NGHIỆM",
        "CHẨN_ĐOÁN",
        "THUỐC",
    }
)


@dataclass(frozen=True, slots=True)
class Phase1VietnameseSupportSpec:
    """Pinned source-model identity and conservative Phase 1 compatibility map."""

    schema_version: str
    run_id: str
    config_path: Path
    run_root: Path
    model: ModelBudgetEntry
    model_source_url: str
    model_permission: str
    model_config: HuggingFaceModelConfig
    stride: int
    confidence_thresholds: dict[str, float]
    compatibility_map: dict[str, tuple[Phase1Label, ...]]

    def __post_init__(self) -> None:
        if self.schema_version != "phase1-vietnamese-support.v1":
            raise ValueError("Unsupported Vietnamese support spec")
        if self.model.parameter_count > 9_000_000_000:
            raise ValueError("Vietnamese support model exceeds the 9B parameter limit")
        if self.stride < 0 or self.stride >= self.model_config.max_length - 2:
            raise ValueError("Vietnamese support stride must fit max_length")
        if set(self.confidence_thresholds) != set(self.compatibility_map):
            raise ValueError("Every compatible source label requires one threshold")
        if any(not 0.0 <= value <= 1.0 for value in self.confidence_thresholds.values()):
            raise ValueError("Vietnamese support thresholds must be between zero and one")
        invalid_targets = {
            target
            for targets in self.compatibility_map.values()
            for target in targets
            if target not in _ALLOWED_PHASE1_LABELS
        }
        if invalid_targets:
            raise ValueError(f"Invalid Phase 1 compatibility labels: {invalid_targets}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize all behavior-bearing model and projection settings."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model": {
                **self.model.to_dict(),
                "source_url": self.model_source_url,
                "permission": self.model_permission,
                "subfolder": self.model_config.subfolder,
            },
            "runtime": {
                "device": self.model_config.device,
                "batch_size": self.model_config.batch_size,
                "max_length": self.model_config.max_length,
                "stride": self.stride,
            },
            "confidence_thresholds": dict(sorted(self.confidence_thresholds.items())),
            "compatibility_map": {
                source: list(targets)
                for source, targets in sorted(self.compatibility_map.items())
            },
        }


def load_phase1_vietnamese_support_spec(
    path: str | Path,
) -> Phase1VietnameseSupportSpec:
    """Load an immutable source-model specification without importing ML dependencies."""

    config_path = Path(path).resolve()
    raw = read_yaml(config_path)
    run_root = _resolve(config_path.parent, _required_string(raw, "run_root"))
    model_raw = _mapping(raw, "model")
    runtime = _mapping(raw, "runtime")
    thresholds = _mapping(raw, "confidence_thresholds")
    compatibility = _mapping(raw, "compatibility_map")
    revision = _required_string(model_raw, "revision")
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise ValueError("Vietnamese support revision must be a full commit SHA")
    roles = model_raw.get("roles")
    if not isinstance(roles, list):
        raise ValueError("Vietnamese support model roles must be a list")
    model = ModelBudgetEntry(
        artifact_id=_required_string(model_raw, "artifact_id"),
        model_id=_required_string(model_raw, "model_id"),
        revision=revision,
        parameter_count=int(model_raw["parameter_count"]),
        kind=str(model_raw.get("kind", "base")),  # type: ignore[arg-type]
        roles=tuple(sorted(str(value) for value in roles)),
    )
    model_config = HuggingFaceModelConfig(
        model_id=model.model_id,
        revision=model.revision,
        subfolder=_optional_string(model_raw, "subfolder"),
        device=str(runtime.get("device", "cuda")),
        batch_size=int(runtime.get("batch_size", 16)),
        max_length=int(runtime.get("max_length", 512)),
    )
    return Phase1VietnameseSupportSpec(
        schema_version=_required_string(raw, "schema_version"),
        run_id=_required_string(raw, "run_id"),
        config_path=config_path,
        run_root=run_root,
        model=model,
        model_source_url=_required_string(model_raw, "source_url"),
        model_permission=_required_string(model_raw, "permission"),
        model_config=model_config,
        stride=int(runtime.get("stride", 64)),
        confidence_thresholds={
            str(label): float(value) for label, value in thresholds.items()
        },
        compatibility_map={
            str(source): _phase1_labels(targets)
            for source, targets in compatibility.items()
        },
    )


def build_phase1_vietnamese_model_support(
    spec: Phase1VietnameseSupportSpec,
    *,
    documents_path: str | Path,
    expected_source_archive_sha256: str,
    output_dir: str | Path,
    expected_document_count: int = 100,
) -> dict[str, Any]:
    """Run the source model and write support-only Phase 1-compatible proposals."""

    documents = load_phase1_round2_documents(
        load_documents(documents_path),
        expected_archive_sha256=expected_source_archive_sha256,
        expected_count=expected_document_count,
    )
    adapter = HuggingFaceSourceTokenClassifierAdapter(
        spec.model_config,
        stride=spec.stride,
        confidence_thresholds=spec.confidence_thresholds,
        default_confidence_threshold=1.0,
    )
    destination = Path(output_dir)
    proposal_dir = destination / "support"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    trace: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for document in documents:
        source_entities = adapter.extract(document.text)
        rows = project_vietnamese_support_rows(
            document.text,
            source_entities,
            compatibility_map=spec.compatibility_map,
        )
        (proposal_dir / f"{document.document_id}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        counters["source_entities"] += len(source_entities)
        counters["support_rows"] += len(rows)
        for entity in source_entities:
            counters[f"source_label.{entity.source_label}"] += 1
        trace.append(
            {
                "document_id": document.document_id,
                "source_entity_count": len(source_entities),
                "support_row_count": len(rows),
            }
        )
    trace_sha256 = write_jsonl(destination / "trace.jsonl", trace)
    manifest = {
        "schema_version": "phase1-vietnamese-model-support.v1",
        "run_spec": spec.to_dict(),
        "inputs": {
            "documents": {
                "path": str(documents_path),
                "sha256": sha256_file(documents_path),
                "source_archive_sha256": expected_source_archive_sha256,
            },
            "config": {
                "path": str(spec.config_path),
                "sha256": sha256_file(spec.config_path),
            },
        },
        "outputs": {
            "support_dir": str(proposal_dir),
            "trace_sha256": trace_sha256,
        },
        "counters": dict(sorted(counters.items())),
        "policy": {
            "direct_output_allowed": False,
            "qwen_confirmation_required": True,
            "source_labels_preserved_in_trace": True,
            "assertions": [],
            "candidates": [],
        },
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def project_vietnamese_support_rows(
    source_text: str,
    entities: Sequence[ProjectedSourceEntity],
    *,
    compatibility_map: Mapping[str, Sequence[Phase1Label]],
) -> list[dict[str, Any]]:
    """Expand broad source labels into candidates that still require Qwen confirmation."""

    rows: list[dict[str, Any]] = []
    for entity in entities:
        start, end = entity.span
        text = source_text[start:end]
        if not text:
            raise ValueError("Vietnamese support entity has an empty raw span")
        targets = compatibility_map.get(entity.source_label, ())
        for target in targets:
            if target not in _ALLOWED_PHASE1_LABELS:
                raise ValueError(f"Unsupported Phase 1 support target: {target}")
            rows.append(
                {
                    "text": text,
                    "type": target,
                    "assertions": [],
                    "candidates": [],
                    "position": [start, end],
                    "confidence": entity.confidence,
                    "source_label": entity.source_label,
                    "support_only": True,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            int(row["position"][0]),
            int(row["position"][1]),
            str(row["type"]),
        ),
    )


def _phase1_labels(value: object) -> tuple[Phase1Label, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("Compatibility-map values must be non-empty lists")
    labels = tuple(str(item) for item in value)
    if len(labels) != len(set(labels)):
        raise ValueError("Compatibility-map labels must be unique")
    return labels  # type: ignore[return-value]


def _mapping(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return {str(name): item for name, item in value.items()}


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()
