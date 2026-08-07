"""Verified one-command development calibration for the Phase 1 model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from clingrounder.benchmarks.phase1 import model_runtime
from clingrounder.benchmarks.phase1.model_runtime import (
    run_phase1_model_calibration,
)
from clingrounder.benchmarks.phase1.model_selection import (
    Phase1ModelSelectionConfig,
)
from clingrounder.cli.parser import build_parser
from clingrounder.pipeline import ResolvedPipelineConfig
from clingrounder.schema.annotation import EntityAnnotation
from clingrounder.schema.output import ClinicalPrediction
from clingrounder.schema.types import EntityType
from clingrounder.training import fingerprint_model_directory
from clingrounder.utils.hashing import sha256_file, sha256_text
from clingrounder.utils.text import normalize_for_match

_LOCK_TEXT = "version = 1\n"
_REVISION = "a" * 40


def test_verified_model_calibration_writes_hashed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_spec_path, model_dir = _write_verified_model_run(tmp_path)
    pipeline_path = _write_model_pipeline(tmp_path, run_spec_path, model_dir)
    selection = _selection_fixture(tmp_path)

    class FakeRunner:
        def process_document(self, document: object) -> ClinicalPrediction:
            document_id = str(getattr(document, "document_id"))
            text = str(getattr(document, "text"))
            if document_id == "1":
                entities = [_entity("đau", EntityType.SYMPTOM, 0.7)]
            else:
                entities = [_entity("hen", EntityType.DISEASE, 0.8)]
            return ClinicalPrediction.from_text(
                document_id,
                text,
                entities,
                [],
                pipeline_version="fixture",
            )

    monkeypatch.setattr(
        model_runtime.PipelineFactory,
        "from_config",
        classmethod(lambda _cls, _config: FakeRunner()),
    )

    report = run_phase1_model_calibration(
        pipeline_path,
        tmp_path / "runs",
        selection_config=selection,
    )

    run_dir = Path(report["run_dir"])
    assert report["status"] == "complete"
    assert report["holdout_status"] == "sealed"
    assert report["document_count"] == 2
    assert (run_dir / "development_predictions.jsonl").is_file()
    assert (run_dir / "calibration.json").is_file()
    calibrated = yaml.safe_load(
        (run_dir / "pipeline_calibrated.yaml").read_text(encoding="utf-8")
    )
    assert set(
        calibrated["models"]["entity_extractor"]["confidence_thresholds"]
    ) == {"DISEASE", "DRUG", "LAB_RESULT", "LAB_TEST", "SYMPTOM"}
    assert calibrated["provenance"]["phase1_model_calibration"]["holdout_status"] == (
        "sealed"
    )
    reloaded = ResolvedPipelineConfig.load(run_dir / "pipeline_calibrated.yaml")
    assert reloaded.factory_config.models.entity_extractor is not None
    assert reloaded.factory_config.models.entity_extractor.model_id == str(model_dir)
    assert reloaded.payload["models"]["entity_extractor"]["run_spec"] == str(
        run_spec_path
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["model_artifact"]["fingerprint"] == fingerprint_model_directory(
        model_dir
    )


def test_model_calibration_cli_has_only_pipeline_and_output_surface() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "model-data",
            "calibrate",
            "--output-dir",
            "outputs/models/calibration",
        ]
    )

    assert args.handler == "benchmark_phase1_model_data_calibrate"
    assert args.pipeline_config.endswith("phase1-five-type-model-only.yaml")
    assert not hasattr(args, "pred")
    assert not hasattr(args, "thresholds")


def _write_verified_model_run(tmp_path: Path) -> tuple[Path, Path]:
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(_LOCK_TEXT, encoding="utf-8")
    dataset_path = tmp_path / "spans.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")
    dataset_manifest = tmp_path / "manifest.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    run_spec_path = tmp_path / "run.yaml"
    run_spec_path.write_text(
        f"""\
schema_version: token-classifier-run.v2
run_id: fixture-five-type
run_root: .
environment:
  lock_path: uv.lock
  lock_sha256: {sha256_text(_LOCK_TEXT)}
dataset:
  path: spans.jsonl
  manifest: manifest.json
  train_split: train
  evaluation_split: development
model:
  model_id: local/model
  revision: {_REVISION}
  source_url: https://example.test/model
  license: MIT
training:
  output_dir: outputs/model
  max_length: 512
  stride: 64
  seed: 42
  full_determinism: true
runtime:
  operating_system: linux
  accelerator: cuda
  minimum_devices: 1
  minimum_vram_gib: 16
  minimum_compute_capability: [8, 0]
  precision: bf16
""",
        encoding="utf-8",
    )
    model_dir = tmp_path / "outputs" / "model" / "final-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    run_manifest = {
        "schema_version": "token-classifier-training.v2",
        "model": {
            "output": "outputs/model/final-model",
            "fingerprint": fingerprint_model_directory(model_dir),
            "model_id": "local/model",
            "revision": _REVISION,
            "initialization": {"kind": "huggingface_cache"},
        },
        "run_spec": {
            "sha256": sha256_file(run_spec_path),
            "run_id": "fixture-five-type",
        },
        "environment": {"lock_sha256": sha256_text(_LOCK_TEXT)},
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "gpu_runtime": {"precision": "bf16"},
    }
    (tmp_path / "outputs" / "model" / "run_manifest.json").write_text(
        json.dumps(run_manifest),
        encoding="utf-8",
    )
    return run_spec_path, model_dir


def _write_model_pipeline(
    tmp_path: Path,
    run_spec_path: Path,
    model_dir: Path,
) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "terminology": {
                    "recognition_path": "data/dictionaries/seed_concepts.jsonl",
                    "alias_overlay_path": None,
                    "reviewed_mention_path": None,
                },
                "pipeline": {
                    "enable_context": False,
                    "enable_linking": False,
                    "enable_candidate_reranking": False,
                    "enable_graph_evidence_reranking": False,
                    "enable_entity_kg_validation": False,
                    "enable_relations": False,
                    "enable_relation_kg_validation": False,
                    "candidate_sources": ["exact"],
                },
                "models": {
                    "entity_extractor": {
                        "run_spec": run_spec_path.name,
                        "model_id": str(model_dir.relative_to(tmp_path)),
                        "revision": _REVISION,
                        "device": "cpu",
                        "batch_size": 2,
                        "max_length": 512,
                        "stride": 64,
                        "default_confidence_threshold": 0.0,
                        "confidence_thresholds": {},
                        "combine_with_dictionary": False,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _selection_fixture(tmp_path: Path) -> Phase1ModelSelectionConfig:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    (input_dir / "1.txt").write_text("đau", encoding="utf-8")
    (input_dir / "2.txt").write_text("hen", encoding="utf-8")
    (input_dir / "3.txt").write_text("sealed", encoding="utf-8")
    (gold_dir / "1.json").write_text(
        json.dumps([_phase1_row("đau", "TRIỆU_CHỨNG")], ensure_ascii=False),
        encoding="utf-8",
    )
    (gold_dir / "2.json").write_text(
        json.dumps([_phase1_row("hen", "CHẨN_ĐOÁN")], ensure_ascii=False),
        encoding="utf-8",
    )
    (gold_dir / "3.json").write_text("not-json", encoding="utf-8")
    frozen = tmp_path / "holdout.json"
    frozen.write_text(
        json.dumps({"splits": {"holdout": {"document_ids": ["3"]}}}),
        encoding="utf-8",
    )
    model_split = tmp_path / "split_manifest.json"
    model_split.write_text(
        json.dumps(
            {
                "source_split_manifest_sha256": sha256_file(frozen),
                "source_document_ids": {
                    "train": [],
                    "development": ["1", "2"],
                },
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "model_holdout_baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")
    return Phase1ModelSelectionConfig(
        input_dir=input_dir,
        gold_dir=gold_dir,
        model_split_manifest=model_split,
        frozen_split_manifest=frozen,
        holdout_baseline_artifact=baseline,
        threshold_grid=(0.0, 0.5, 0.75),
    )


def _entity(
    text: str,
    entity_type: EntityType,
    confidence: float,
) -> EntityAnnotation:
    return EntityAnnotation(
        id="M1",
        span=(0, len(text)),
        text=text,
        normalized_text=normalize_for_match(text),
        type=entity_type,
        confidence=confidence,
    )


def _phase1_row(text: str, entity_type: str) -> dict[str, object]:
    row: dict[str, object] = {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "position": [0, len(text)],
    }
    if entity_type == "CHẨN_ĐOÁN":
        row["candidates"] = []
    return row
