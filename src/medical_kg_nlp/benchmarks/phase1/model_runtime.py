"""Verified model inference and development calibration for Phase 1.

The training run specification is the scientific identity. The pipeline YAML only describes how
that returned checkpoint is loaded for model-only inference; it cannot substitute another model
directory or silently enable dictionary, context, linking, relation, or KG stages during threshold
selection.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from medical_kg_nlp.benchmarks.phase1.model_selection import (
    Phase1ModelSelectionConfig,
    calibrate_phase1_model_thresholds,
    infer_phase1_development_predictions,
)
from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.pipeline.config_loader import ResolvedPipelineConfig
from medical_kg_nlp.pipeline.factory import PipelineFactory, PipelineConfig
from medical_kg_nlp.training import (
    TokenClassifierRunSpec,
    load_token_classifier_run_spec,
    verify_token_classifier_run_artifact,
)
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.io import write_jsonl
from medical_kg_nlp.utils.run_output import create_hashed_run_dir

__all__ = ["run_phase1_model_calibration"]


def run_phase1_model_calibration(
    pipeline_config_path: str | Path,
    output_root: str | Path,
    *,
    selection_config: Phase1ModelSelectionConfig | None = None,
) -> dict[str, Any]:
    """Infer the development split, calibrate five thresholds, and persist one hashed run."""

    resolved_pipeline = ResolvedPipelineConfig.load(pipeline_config_path)
    config_path = resolved_pipeline.source_path
    factory_config, run_spec, model_artifact = _verified_factory_config(
        resolved_pipeline,
    )
    active_selection = selection_config or Phase1ModelSelectionConfig(
        model_split_manifest=run_spec.training.dataset_manifest_path.with_name(
            "split_manifest.json"
        )
    )
    run_manifest_path = run_spec.training.output_dir / "run_manifest.json"
    run_output = create_hashed_run_dir(
        output_root,
        label=f"{run_spec.run_id}-development-calibration",
        inputs=(
            config_path,
            run_spec.config_path,
            run_manifest_path,
            active_selection.model_split_manifest,
            active_selection.frozen_split_manifest,
        ),
        resolved_config={
            "selection_split": "development",
            "holdout_status": "sealed",
            "model_fingerprint": model_artifact["fingerprint"],
            "pipeline_version": factory_config.pipeline_version,
        },
        random_seed=run_spec.training.seed,
    )

    # MODEL: one runner keeps a single checkpoint resident while processing all 16 documents.
    runner = PipelineFactory.from_config(factory_config)
    predictions = infer_phase1_development_predictions(
        runner,
        config=active_selection,
    )
    predictions_path = run_output.run_dir / "development_predictions.jsonl"
    write_jsonl(
        predictions_path,
        (
            predictions[document_id].to_json()
            for document_id in _sort_document_ids(predictions)
        ),
    )

    calibration = calibrate_phase1_model_thresholds(
        predictions,
        config=active_selection,
        prediction_path=predictions_path,
    )
    calibration_path = run_output.run_dir / "calibration.json"
    write_json(calibration_path, calibration)
    calibrated_pipeline_path = run_output.run_dir / "pipeline_calibrated.yaml"
    _write_calibrated_pipeline(
        resolved_pipeline,
        calibration,
        calibrated_pipeline_path,
        model_artifact=model_artifact,
        calibration_sha256=sha256_file(calibration_path),
    )

    manifest = json.loads(run_output.manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete",
            "model_artifact": model_artifact,
            "selection": {
                "split": "development",
                "holdout_status": "sealed",
                "document_count": calibration["document_count"],
                "document_ids_sha256": calibration["document_ids_sha256"],
                "selected_thresholds": calibration["selected_thresholds"],
                "metrics": calibration["metrics"],
                "error_counts": calibration["error_counts"],
            },
            "outputs": {
                "predictions": str(predictions_path),
                "predictions_sha256": sha256_file(predictions_path),
                "calibration": str(calibration_path),
                "calibration_sha256": sha256_file(calibration_path),
                "calibrated_pipeline": str(calibrated_pipeline_path),
                "calibrated_pipeline_sha256": sha256_file(calibrated_pipeline_path),
            },
        }
    )
    write_json(run_output.manifest_path, manifest)
    return {
        "status": "complete",
        "run_id": run_output.run_id,
        "run_dir": str(run_output.run_dir),
        "run_manifest": str(run_output.manifest_path),
        "model_fingerprint": model_artifact["fingerprint"],
        "document_count": calibration["document_count"],
        "selected_thresholds": calibration["selected_thresholds"],
        "metrics": calibration["metrics"],
        "error_counts": calibration["error_counts"],
        "holdout_status": "sealed",
    }


def _verified_factory_config(
    resolved_pipeline: ResolvedPipelineConfig,
) -> tuple[PipelineConfig, TokenClassifierRunSpec, dict[str, Any]]:
    models = _mapping(resolved_pipeline.payload.get("models"), "models")
    entity_model = _mapping(models.get("entity_extractor"), "models.entity_extractor")
    raw_run_spec = entity_model.get("run_spec")
    if not isinstance(raw_run_spec, str) or not raw_run_spec.strip():
        raise ValueError("models.entity_extractor.run_spec is required for calibration")

    run_spec = load_token_classifier_run_spec(raw_run_spec)
    artifact = verify_token_classifier_run_artifact(run_spec)
    factory_config = resolved_pipeline.factory_config
    model_config = factory_config.models.entity_extractor
    if model_config is None:
        raise ValueError("Phase 1 calibration requires models.entity_extractor")

    expected_model_dir = (run_spec.training.output_dir / "final-model").resolve()
    configured_model_dir = Path(model_config.model_id).resolve()
    if configured_model_dir != expected_model_dir:
        raise ValueError(
            "models.entity_extractor.model_id must point to the verified final-model directory"
        )
    if model_config.revision != run_spec.training.revision:
        raise ValueError("Entity model revision does not match the training run specification")
    if model_config.max_length != run_spec.training.max_length:
        raise ValueError("Entity model max_length does not match training")
    if factory_config.models.entity_stride != run_spec.training.stride:
        raise ValueError("Entity model stride does not match training")
    if factory_config.models.entity_combine_with_dictionary:
        raise ValueError("Threshold calibration requires model-only entity extraction")
    if (
        factory_config.models.entity_default_confidence_threshold != 0.0
        or factory_config.models.entity_confidence_thresholds
    ):
        raise ValueError("Threshold calibration requires unfiltered model confidences")

    options = factory_config.options
    enabled_downstream = [
        name
        for name, enabled in (
            ("context", options.enable_context),
            ("linking", options.enable_linking),
            ("candidate_reranking", options.enable_candidate_reranking),
            ("graph_evidence", options.enable_graph_evidence_reranking),
            ("entity_kg_validation", options.enable_entity_kg_validation),
            ("relations", options.enable_relations),
            ("relation_kg_validation", options.enable_relation_kg_validation),
        )
        if enabled
    ]
    if enabled_downstream:
        raise ValueError(
            "Threshold calibration pipeline must disable downstream stages: "
            + ", ".join(enabled_downstream)
        )
    # MODEL: the adapter receives an absolute verified directory, so inference is independent of
    # the caller's working directory after configuration has been loaded.
    factory_config = replace(
        factory_config,
        models=replace(
            factory_config.models,
            entity_extractor=replace(
                model_config,
                model_id=str(expected_model_dir),
            ),
        ),
    )
    return factory_config, run_spec, dict(artifact)


def _write_calibrated_pipeline(
    resolved_pipeline: ResolvedPipelineConfig,
    calibration: Mapping[str, Any],
    path: Path,
    *,
    model_artifact: Mapping[str, Any],
    calibration_sha256: str,
) -> None:
    payload = resolved_pipeline.payload_for(path)
    models = _mapping(payload.get("models"), "models")
    entity_model = _mapping(models.get("entity_extractor"), "models.entity_extractor")
    thresholds = calibration.get("selected_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("Calibration report is missing selected_thresholds")
    entity_model["confidence_thresholds"] = {
        str(entity_type): float(threshold)
        for entity_type, threshold in sorted(thresholds.items())
    }
    models["entity_extractor"] = entity_model
    payload["models"] = models
    provenance = _optional_mapping(payload.get("provenance"), "provenance")
    provenance["phase1_model_calibration"] = {
        "selection_split": "development",
        "holdout_status": "sealed",
        "model_fingerprint": str(model_artifact["fingerprint"]),
        "calibration_sha256": calibration_sha256,
    }
    payload["provenance"] = provenance
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _optional_mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _mapping(value, name)


def _sort_document_ids(values: Mapping[str, object]) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            not value.isdigit(),
            int(value) if value.isdigit() else value,
        ),
    )
