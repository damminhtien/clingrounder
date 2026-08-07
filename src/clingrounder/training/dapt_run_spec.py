"""Pinned XLM-R domain-adaptive pretraining run specifications."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.training.run_spec import GPURequirements
from clingrounder.utils.hashing import sha256_file
from clingrounder.utils.io import read_yaml

__all__ = [
    "DaptLaneInput",
    "XlmrDaptRunSpec",
    "XlmrDaptTrainingConfig",
    "inspect_xlmr_dapt_inputs",
    "load_xlmr_dapt_run_spec",
    "verify_xlmr_dapt_run_artifact",
]

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class DaptLaneInput:
    """One verified lane from an immutable DAPT corpus manifest."""

    lane_id: str
    kind: str
    path: Path
    sha256: str
    record_count: int
    sampling_weight: float

    def __post_init__(self) -> None:
        if not self.lane_id.strip():
            raise ValueError("DAPT lane_id must be non-empty")
        if self.kind not in {"open_unlabeled", "round2_unlabeled"}:
            raise ValueError(f"Unsupported DAPT lane kind: {self.kind}")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("DAPT lane SHA-256 is invalid")
        if self.record_count < 1:
            raise ValueError("DAPT training lanes cannot be empty")
        if self.sampling_weight <= 0:
            raise ValueError("DAPT lane sampling weight must be positive")


@dataclass(frozen=True, slots=True)
class XlmrDaptTrainingConfig:
    """Framework-neutral inputs and hyperparameters for joint DAPT."""

    model_id: str
    revision: str
    cache_dir: Path | None
    output_dir: Path
    lanes: tuple[DaptLaneInput, ...]
    corpus_manifest_path: Path
    synonym_pairs_path: Path
    synonym_manifest_path: Path
    synonym_source_fingerprints: tuple[tuple[Path, str], ...]
    max_length: int = 256
    mlm_probability: float = 0.15
    mlm_batch_size: int = 8
    contrastive_batch_size: int = 16
    contrastive_weight: float = 0.2
    contrastive_temperature: float = 0.05
    max_steps: int = 10_000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    checkpoint_interval: int = 1_000
    preprocessing_workers: int = 2
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("DAPT model_id must be non-empty")
        if _COMMIT_SHA.fullmatch(self.revision) is None:
            raise ValueError("DAPT model revision must be a full commit SHA")
        if not self.lanes:
            raise ValueError("DAPT requires at least one MLM lane")
        if not self.synonym_source_fingerprints:
            raise ValueError("DAPT requires pinned terminology sources")
        source_paths = [path for path, _ in self.synonym_source_fingerprints]
        if len(source_paths) != len(set(source_paths)):
            raise ValueError("DAPT terminology source paths must be unique")
        for source_path, source_sha256 in self.synonym_source_fingerprints:
            if _SHA256.fullmatch(source_sha256) is None:
                raise ValueError(
                    f"DAPT terminology source SHA-256 is invalid: {source_path}"
                )
        if self.max_length < 32:
            raise ValueError("DAPT max_length must be at least 32")
        if not 0.0 < self.mlm_probability < 1.0:
            raise ValueError("DAPT mlm_probability must be in (0, 1)")
        if self.mlm_batch_size < 1 or self.contrastive_batch_size < 2:
            raise ValueError("DAPT batch sizes are invalid")
        if self.contrastive_weight <= 0:
            raise ValueError("DAPT contrastive_weight must be positive")
        if self.contrastive_temperature <= 0:
            raise ValueError("DAPT contrastive_temperature must be positive")
        if self.max_steps < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("DAPT training steps must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("DAPT optimizer values are invalid")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("DAPT warmup_ratio must be in [0, 1)")
        if self.checkpoint_interval < 1 or self.preprocessing_workers < 1:
            raise ValueError("DAPT checkpoint/workers values must be positive")


@dataclass(frozen=True, slots=True)
class XlmrDaptRunSpec:
    """Portable, immutable identity for one XLM-R DAPT experiment."""

    schema_version: str
    run_id: str
    config_path: Path
    run_root: Path
    training: XlmrDaptTrainingConfig
    runtime: GPURequirements
    environment_lock_path: Path
    environment_lock_sha256: str
    model_source_url: str
    model_license: str

    def __post_init__(self) -> None:
        if self.schema_version != "xlmr-dapt-run.v1":
            raise ValueError("Unsupported XLM-R DAPT run schema")
        if not self.run_id.strip():
            raise ValueError("DAPT run_id must be non-empty")
        if self.runtime.operating_system != "linux" or self.runtime.accelerator != "cuda":
            raise ValueError("DAPT run requires Linux/CUDA")
        if self.runtime.precision not in {"bf16", "fp16"}:
            raise ValueError("DAPT runtime precision must be bf16 or fp16")
        if _SHA256.fullmatch(self.environment_lock_sha256) is None:
            raise ValueError("DAPT environment lock hash is invalid")
        if sha256_file(self.environment_lock_path) != self.environment_lock_sha256:
            raise ValueError("DAPT environment lock SHA-256 mismatch")
        self.relative_path(self.config_path)
        self.relative_path(self.training.output_dir)
        self.relative_path(self.training.corpus_manifest_path)
        self.relative_path(self.training.synonym_pairs_path)
        self.relative_path(self.training.synonym_manifest_path)
        for source_path, _ in self.training.synonym_source_fingerprints:
            self.relative_path(source_path)
        for lane in self.training.lanes:
            self.relative_path(lane.path)

    @property
    def config_relative_path(self) -> str:
        return self.relative_path(self.config_path)

    @property
    def prefetch_command(self) -> tuple[str, ...]:
        command: tuple[str, ...] = (
            "hf",
            "download",
            self.training.model_id,
            "--revision",
            self.training.revision,
        )
        if self.training.cache_dir is not None:
            command += (
                "--cache-dir",
                self.relative_path(self.training.cache_dir),
            )
        return command

    def relative_path(self, path: str | Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.run_root).as_posix()
        except ValueError as error:
            raise ValueError(f"DAPT path escapes run_root: {path}") from error


def load_xlmr_dapt_run_spec(path: str | Path) -> XlmrDaptRunSpec:
    """Load and verify one DAPT YAML plus all derived dataset manifests."""

    config_path = Path(path).resolve()
    raw = read_yaml(config_path)
    run_root = _resolve(config_path.parent, _required_string(raw, "run_root"))
    corpus_raw = _mapping(raw, "corpus")
    synonym_raw = _mapping(raw, "synonym_pairs")
    model_raw = _mapping(raw, "model")
    training_raw = _mapping(raw, "training")
    runtime_raw = _mapping(raw, "runtime")
    environment_raw = _mapping(raw, "environment")
    corpus_manifest_path = _resolve(
        run_root,
        _required_string(corpus_raw, "manifest"),
    )
    corpus_manifest = _json_mapping(corpus_manifest_path, "DAPT corpus manifest")
    lanes = _load_lanes(corpus_manifest, run_root)
    synonym_pairs_path = _resolve(
        run_root,
        _required_string(synonym_raw, "path"),
    )
    synonym_manifest_path = _resolve(
        run_root,
        _required_string(synonym_raw, "manifest"),
    )
    synonym_source_fingerprints = _synonym_sources(
        synonym_raw,
        run_root,
    )
    _verify_synonym_pairs(
        synonym_pairs_path,
        synonym_manifest_path,
        expected_sources=synonym_source_fingerprints,
        run_root=run_root,
    )
    capability = runtime_raw.get("minimum_compute_capability", [8, 0])
    if not isinstance(capability, list) or len(capability) != 2:
        raise ValueError("minimum_compute_capability must be [major, minor]")
    training = XlmrDaptTrainingConfig(
        model_id=_required_string(model_raw, "model_id"),
        revision=_required_string(model_raw, "revision"),
        cache_dir=(
            None
            if training_raw.get("cache_dir") is None
            else _resolve(run_root, str(training_raw["cache_dir"]))
        ),
        output_dir=_resolve(
            run_root,
            _required_string(training_raw, "output_dir"),
        ),
        lanes=lanes,
        corpus_manifest_path=corpus_manifest_path,
        synonym_pairs_path=synonym_pairs_path,
        synonym_manifest_path=synonym_manifest_path,
        synonym_source_fingerprints=synonym_source_fingerprints,
        max_length=int(training_raw.get("max_length", 256)),
        mlm_probability=float(training_raw.get("mlm_probability", 0.15)),
        mlm_batch_size=int(training_raw.get("mlm_batch_size", 8)),
        contrastive_batch_size=int(
            training_raw.get("contrastive_batch_size", 16)
        ),
        contrastive_weight=float(training_raw.get("contrastive_weight", 0.2)),
        contrastive_temperature=float(
            training_raw.get("contrastive_temperature", 0.05)
        ),
        max_steps=int(training_raw.get("max_steps", 10_000)),
        gradient_accumulation_steps=int(
            training_raw.get("gradient_accumulation_steps", 1)
        ),
        learning_rate=float(training_raw.get("learning_rate", 5e-5)),
        weight_decay=float(training_raw.get("weight_decay", 0.01)),
        warmup_ratio=float(training_raw.get("warmup_ratio", 0.06)),
        seed=int(training_raw.get("seed", 42)),
        checkpoint_interval=int(
            training_raw.get("checkpoint_interval", 1_000)
        ),
        preprocessing_workers=int(
            training_raw.get("preprocessing_workers", 2)
        ),
        local_files_only=bool(training_raw.get("local_files_only", True)),
    )
    return XlmrDaptRunSpec(
        schema_version=_required_string(raw, "schema_version"),
        run_id=_required_string(raw, "run_id"),
        config_path=config_path,
        run_root=run_root,
        training=training,
        runtime=GPURequirements(
            operating_system=str(runtime_raw.get("operating_system", "linux")),
            accelerator=str(runtime_raw.get("accelerator", "cuda")),
            minimum_devices=int(runtime_raw.get("minimum_devices", 1)),
            minimum_vram_gib=float(runtime_raw.get("minimum_vram_gib", 16)),
            minimum_compute_capability=(int(capability[0]), int(capability[1])),
            precision=str(runtime_raw.get("precision", "bf16")),
        ),
        environment_lock_path=_resolve(
            run_root,
            _required_string(environment_raw, "lock_path"),
        ),
        environment_lock_sha256=_required_string(
            environment_raw,
            "lock_sha256",
        ),
        model_source_url=_required_string(model_raw, "source_url"),
        model_license=_required_string(model_raw, "license"),
    )


def inspect_xlmr_dapt_inputs(spec: XlmrDaptRunSpec) -> dict[str, Any]:
    """Re-verify all bytes and summarize objective-isolated training inputs."""

    corpus_manifest = _json_mapping(
        spec.training.corpus_manifest_path,
        "DAPT corpus manifest",
    )
    _verify_round2_policy(corpus_manifest, spec.training.lanes)
    lane_reports = []
    for lane in spec.training.lanes:
        actual = sha256_file(lane.path)
        if actual != lane.sha256:
            raise ValueError(
                f"DAPT lane hash mismatch for {lane.lane_id}: "
                f"expected {lane.sha256}, got {actual}"
            )
        lane_reports.append(
            {
                "lane_id": lane.lane_id,
                "kind": lane.kind,
                "path": spec.relative_path(lane.path),
                "sha256": actual,
                "record_count": lane.record_count,
                "sampling_weight": lane.sampling_weight,
                "objectives": ["masked_language_modeling"],
            }
        )
    pair_manifest = _verify_synonym_pairs(
        spec.training.synonym_pairs_path,
        spec.training.synonym_manifest_path,
        expected_sources=spec.training.synonym_source_fingerprints,
        run_root=spec.run_root,
    )
    return {
        "corpus_manifest": {
            "path": spec.relative_path(spec.training.corpus_manifest_path),
            "sha256": sha256_file(spec.training.corpus_manifest_path),
        },
        "lanes": lane_reports,
        "synonym_pairs": {
            "path": spec.relative_path(spec.training.synonym_pairs_path),
            "sha256": pair_manifest["dataset_sha256"],
            "record_count": int(pair_manifest["record_count"]),
            "round2_included": False,
            "sources": [
                {
                    "path": spec.relative_path(path),
                    "sha256": source_sha256,
                }
                for path, source_sha256 in spec.training.synonym_source_fingerprints
            ],
        },
        "round2_policy": corpus_manifest["round2_unlabeled_policy"],
    }


def verify_xlmr_dapt_run_artifact(
    spec: XlmrDaptRunSpec,
) -> dict[str, Any]:
    """Verify a promotable DAPT checkpoint without loading model dependencies.

    The saved encoder is reusable by later NER and retrieval jobs, so this gate
    binds it to the exact run specification, input bytes, objective separation,
    dependency lock, and GPU precision that produced it.
    """

    from clingrounder.training.huggingface_token_classifier import (
        fingerprint_model_directory,
    )

    manifest_path = spec.training.output_dir / "run_manifest.json"
    manifest = _json_mapping(manifest_path, "XLM-R DAPT run manifest")
    if manifest.get("schema_version") != "xlmr-dapt-artifact.v1":
        raise ValueError("Unsupported XLM-R DAPT artifact schema")
    training = _as_mapping(manifest.get("training"), "DAPT training metadata")
    if (
        manifest.get("promotion_eligible") is not True
        or manifest.get("purpose") != "training"
        or training.get("smoke") is not False
    ):
        raise ValueError("Smoke or non-promotable DAPT artifact cannot be selected")
    if training.get("global_step") != spec.training.max_steps:
        raise ValueError("DAPT artifact did not complete the configured training steps")
    source_control = _as_mapping(
        manifest.get("source_control"),
        "DAPT source-control provenance",
    )
    if (
        _COMMIT_SHA.fullmatch(str(source_control.get("git_commit", ""))) is None
        or source_control.get("git_dirty") is not False
    ):
        raise ValueError("DAPT artifact requires a clean committed source revision")

    expected_model_dir = spec.training.output_dir / "final-model"
    model = _as_mapping(manifest.get("model"), "DAPT model metadata")
    if model.get("output") != spec.relative_path(expected_model_dir):
        raise ValueError("DAPT model output does not match the run specification")
    if model.get("model_id") != spec.training.model_id:
        raise ValueError("DAPT model ID does not match the run specification")
    if model.get("revision") != spec.training.revision:
        raise ValueError("DAPT model revision does not match the run specification")
    expected_fingerprint = model.get("fingerprint")
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise ValueError("DAPT model fingerprint is absent")
    actual_fingerprint = fingerprint_model_directory(expected_model_dir)
    if actual_fingerprint != expected_fingerprint:
        raise ValueError("DAPT model fingerprint does not match saved model bytes")

    run_spec = _as_mapping(manifest.get("run_spec"), "DAPT run-spec provenance")
    if run_spec.get("run_id") != spec.run_id:
        raise ValueError("DAPT artifact run ID does not match")
    if run_spec.get("sha256") != sha256_file(spec.config_path):
        raise ValueError("DAPT artifact run-spec SHA-256 does not match")
    environment = _as_mapping(
        manifest.get("environment"),
        "DAPT environment provenance",
    )
    if environment.get("lock_sha256") != spec.environment_lock_sha256:
        raise ValueError("DAPT artifact environment lock SHA-256 does not match")
    gpu_runtime = _as_mapping(
        manifest.get("gpu_runtime"),
        "DAPT GPU provenance",
    )
    if gpu_runtime.get("precision") != spec.runtime.precision:
        raise ValueError("DAPT artifact GPU precision does not match")

    input_report = inspect_xlmr_dapt_inputs(spec)
    if manifest.get("input_verification") != input_report:
        raise ValueError("DAPT artifact input provenance does not match current bytes")
    _verify_artifact_objectives(
        manifest,
        expected_lane_ids=[lane.lane_id for lane in spec.training.lanes],
    )
    return {
        "status": "verified",
        "manifest": spec.relative_path(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model": spec.relative_path(expected_model_dir),
        "model_id": spec.training.model_id,
        "revision": spec.training.revision,
        "fingerprint": actual_fingerprint,
        "gpu_runtime": dict(gpu_runtime),
        "round2_policy": input_report["round2_policy"],
    }


def _verify_artifact_objectives(
    manifest: dict[str, Any],
    *,
    expected_lane_ids: list[str],
) -> None:
    """Enforce that Round 2 stayed in MLM and out of contrastive training."""

    objectives = _as_mapping(manifest.get("objectives"), "DAPT objectives")
    mlm = _as_mapping(
        objectives.get("masked_language_modeling"),
        "DAPT MLM objective",
    )
    if mlm.get("lanes") != expected_lane_ids:
        raise ValueError("DAPT artifact MLM lanes do not match the run specification")
    contrastive = _as_mapping(
        objectives.get("synonym_contrastive"),
        "DAPT synonym-contrastive objective",
    )
    if contrastive.get("round2_included") is not False:
        raise ValueError("Round 2 cannot enter synonym-contrastive DAPT")


def _load_lanes(
    manifest: dict[str, Any],
    run_root: Path,
) -> tuple[DaptLaneInput, ...]:
    if manifest.get("schema_version") != "xlmr-dapt-corpus.v1":
        raise ValueError("Unsupported DAPT corpus manifest")
    values = manifest.get("lanes")
    if not isinstance(values, list) or not values:
        raise ValueError("DAPT corpus manifest requires lanes")
    lanes = []
    for value in values:
        raw = _as_mapping(value, "DAPT lane")
        lanes.append(
            DaptLaneInput(
                lane_id=_required_string(raw, "lane_id"),
                kind=_required_string(raw, "kind"),
                path=_resolve(run_root, _required_string(raw, "path")),
                sha256=_required_string(raw, "sha256"),
                record_count=int(raw["record_count"]),
                sampling_weight=float(raw["sampling_weight"]),
            )
        )
    output = tuple(lanes)
    _verify_round2_policy(manifest, output)
    return output


def _verify_round2_policy(
    manifest: dict[str, Any],
    lanes: tuple[DaptLaneInput, ...],
) -> None:
    policy = _as_mapping(
        manifest.get("round2_unlabeled_policy"),
        "round2_unlabeled_policy",
    )
    expected_ids = sorted(
        lane.lane_id for lane in lanes if lane.kind == "round2_unlabeled"
    )
    if policy.get("lane_ids") != expected_ids:
        raise ValueError("DAPT Round 2 lane provenance mismatch")
    if (
        policy.get("supervision") != "none"
        or policy.get("allowed_objectives") != ["masked_language_modeling"]
    ):
        raise ValueError("Round 2 DAPT lane must remain unlabeled and MLM-only")
    forbidden = policy.get("forbidden_objectives")
    if not isinstance(forbidden, list) or "synonym_contrastive" not in forbidden:
        raise ValueError("Round 2 DAPT policy must forbid synonym contrastive use")


def _verify_synonym_pairs(
    pairs_path: Path,
    manifest_path: Path,
    *,
    expected_sources: tuple[tuple[Path, str], ...],
    run_root: Path,
) -> dict[str, Any]:
    if not pairs_path.is_file() or not manifest_path.is_file():
        raise ValueError("DAPT synonym pair artifacts are absent")
    manifest = _json_mapping(manifest_path, "synonym-pair manifest")
    if manifest.get("schema_version") != "terminology-synonym-pairs.v1":
        raise ValueError("Unsupported synonym-pair manifest")
    if manifest.get("dataset_sha256") != sha256_file(pairs_path):
        raise ValueError("DAPT synonym-pair SHA-256 mismatch")
    if int(manifest.get("record_count", 0)) < 1:
        raise ValueError("DAPT synonym-pair dataset is empty")
    sources = manifest.get("source_fingerprints")
    if not isinstance(sources, dict):
        raise ValueError("Synonym-pair source fingerprints are absent")
    expected = {
        path.resolve().relative_to(run_root).as_posix(): source_sha256
        for path, source_sha256 in expected_sources
    }
    # Builders record run-root-relative paths. Normalize separators without
    # weakening exact source-set equality.
    actual = {
        Path(str(path)).as_posix(): str(source_sha256)
        for path, source_sha256 in sources.items()
    }
    if actual != expected:
        raise ValueError("DAPT synonym-pair source provenance differs from the run spec")
    for source_path, expected_sha256 in expected_sources:
        relative_source = source_path.resolve().relative_to(run_root).as_posix()
        if "round2" in relative_source.lower():
            raise ValueError("Round 2 cannot supply DAPT synonym pairs")
        if not source_path.is_file():
            raise ValueError(f"DAPT terminology source is absent: {source_path}")
        if sha256_file(source_path) != expected_sha256:
            raise ValueError(
                f"DAPT terminology source SHA-256 mismatch: {source_path}"
            )
    return manifest


def _synonym_sources(
    raw: dict[str, Any],
    run_root: Path,
) -> tuple[tuple[Path, str], ...]:
    values = raw.get("sources")
    if not isinstance(values, list) or not values:
        raise ValueError("synonym_pairs.sources must be a non-empty list")
    output: list[tuple[Path, str]] = []
    for value in values:
        source = _as_mapping(value, "synonym-pair source")
        output.append(
            (
                _resolve(run_root, _required_string(source, "path")),
                _required_string(source, "sha256"),
            )
        )
    return tuple(sorted(output, key=lambda item: item[0].as_posix()))


def _mapping(raw: dict[str, Any], field: str) -> dict[str, Any]:
    return _as_mapping(raw.get(field), field)


def _as_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required_string(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _json_mapping(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is invalid JSON: {path}") from error
    return _as_mapping(value, field)
