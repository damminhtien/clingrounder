"""Contracts for joint XLM-R DAPT and objective-isolated provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clingrounder.cli.parser import build_parser
from clingrounder.training.dapt_run_spec import (
    inspect_xlmr_dapt_inputs,
    load_xlmr_dapt_run_spec,
    verify_xlmr_dapt_run_artifact,
)
from clingrounder.training.huggingface_token_classifier import (
    fingerprint_model_directory,
)
from clingrounder.training.xlmr_dapt import (
    xlmr_dapt_input_provenance,
)


def test_dapt_run_keeps_round2_in_mlm_and_out_of_contrastive(
    tmp_path: Path,
) -> None:
    config = _write_run_fixture(tmp_path)

    spec = load_xlmr_dapt_run_spec(config)
    report = inspect_xlmr_dapt_inputs(spec)
    provenance = xlmr_dapt_input_provenance(
        spec.training,
        manifest_root=spec.run_root,
    )

    assert {
        (row["lane_id"], row["kind"], tuple(row["objectives"]))
        for row in report["lanes"]
    } == {
        ("open", "open_unlabeled", ("masked_language_modeling",)),
        ("round2", "round2_unlabeled", ("masked_language_modeling",)),
    }
    assert report["synonym_pairs"]["round2_included"] is False
    assert provenance["synonym_pairs"]["round2_included"] is False
    round2 = next(
        row for row in provenance["mlm_lanes"] if row["kind"] == "round2_unlabeled"
    )
    assert round2["supervision"] == "none"
    assert round2["objective"] == "masked_language_modeling"


def test_dapt_run_rejects_changed_lane_or_terminology_bytes(
    tmp_path: Path,
) -> None:
    config = _write_run_fixture(tmp_path)
    spec = load_xlmr_dapt_run_spec(config)
    spec.training.lanes[0].path.write_text('{"text":"changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="lane hash mismatch"):
        inspect_xlmr_dapt_inputs(spec)

    config = _write_run_fixture(tmp_path / "second")
    terminology = tmp_path / "second/data/terminology.jsonl"
    terminology.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="terminology source SHA-256 mismatch"):
        load_xlmr_dapt_run_spec(config)


def test_dapt_run_rejects_round2_as_synonym_source(tmp_path: Path) -> None:
    config = _write_run_fixture(
        tmp_path,
        terminology_relative_path="data/round2-terminology.jsonl",
    )

    with pytest.raises(ValueError, match="Round 2 cannot supply"):
        load_xlmr_dapt_run_spec(config)


def test_dapt_cli_commands_are_discoverable(tmp_path: Path) -> None:
    parser = build_parser()

    inspect_args = parser.parse_args(
        ["model", "inspect-xlmr-dapt-run", "--config", "dapt.yaml"]
    )
    train_args = parser.parse_args(
        [
            "model",
            "train-xlmr-dapt-run",
            "--config",
            "dapt.yaml",
            "--max-steps",
            "1",
            "--output-dir",
            "outputs/smoke/dapt",
        ]
    )

    assert inspect_args.handler == "model_inspect_xlmr_dapt_run"
    assert train_args.handler == "model_train_xlmr_dapt_run"
    assert train_args.max_steps == 1

    spec = load_xlmr_dapt_run_spec(_write_run_fixture(tmp_path))
    assert spec.prefetch_command[-2:] == (
        "--cache-dir",
        ".cache/model-training",
    )


def test_dapt_artifact_verifier_binds_model_and_objective_provenance(
    tmp_path: Path,
) -> None:
    config = _write_run_fixture(tmp_path)
    spec = load_xlmr_dapt_run_spec(config)
    _write_artifact_fixture(spec)

    report = verify_xlmr_dapt_run_artifact(spec)

    assert report["status"] == "verified"
    assert report["model"] == "outputs/model/final-model"

    manifest_path = spec.training.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objectives"]["synonym_contrastive"]["round2_included"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Round 2 cannot enter"):
        verify_xlmr_dapt_run_artifact(spec)


def test_dapt_artifact_verifier_rejects_tampered_model(
    tmp_path: Path,
) -> None:
    config = _write_run_fixture(tmp_path)
    spec = load_xlmr_dapt_run_spec(config)
    _write_artifact_fixture(spec)
    (spec.training.output_dir / "final-model/model.safetensors").write_text(
        "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        verify_xlmr_dapt_run_artifact(spec)


def _write_artifact_fixture(spec: object) -> None:
    # Tests keep the fixture independent from the optional Torch runtime.
    training = spec.training
    model_dir = training.output_dir / "final-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")
    manifest = {
        "schema_version": "xlmr-dapt-artifact.v1",
        "model": {
            "model_id": training.model_id,
            "revision": training.revision,
            "output": spec.relative_path(model_dir),
            "fingerprint": fingerprint_model_directory(model_dir),
        },
        "objectives": {
            "masked_language_modeling": {
                "lanes": [lane.lane_id for lane in training.lanes],
            },
            "synonym_contrastive": {"round2_included": False},
        },
        "training": {
            "global_step": training.max_steps,
            "smoke": False,
        },
        "run_spec": {
            "run_id": spec.run_id,
            "sha256": _sha256(spec.config_path),
        },
        "environment": {
            "lock_sha256": spec.environment_lock_sha256,
        },
        "gpu_runtime": {"precision": spec.runtime.precision},
        "input_verification": inspect_xlmr_dapt_inputs(spec),
        "source_control": {
            "git_commit": "a" * 40,
            "git_dirty": False,
        },
        "purpose": "training",
        "promotion_eligible": True,
    }
    (training.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _write_run_fixture(
    root: Path,
    *,
    terminology_relative_path: str = "data/terminology.jsonl",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "uv.lock"
    lock.write_text("locked\n", encoding="utf-8")
    open_lane = root / "derived/open.jsonl"
    round2_lane = root / "derived/round2.jsonl"
    open_lane.parent.mkdir(parents=True)
    open_lane.write_text('{"text":"bệnh nhân ho"}\n', encoding="utf-8")
    round2_lane.write_text('{"text":"round 2 private"}\n', encoding="utf-8")
    corpus_manifest = root / "derived/corpus-manifest.json"
    corpus_manifest.write_text(
        json.dumps(
            {
                "schema_version": "xlmr-dapt-corpus.v1",
                "lanes": [
                    {
                        "lane_id": "open",
                        "kind": "open_unlabeled",
                        "path": "derived/open.jsonl",
                        "sha256": _sha256(open_lane),
                        "record_count": 1,
                        "sampling_weight": 1.0,
                    },
                    {
                        "lane_id": "round2",
                        "kind": "round2_unlabeled",
                        "path": "derived/round2.jsonl",
                        "sha256": _sha256(round2_lane),
                        "record_count": 1,
                        "sampling_weight": 0.25,
                    },
                ],
                "round2_unlabeled_policy": {
                    "lane_ids": ["round2"],
                    "supervision": "none",
                    "allowed_objectives": ["masked_language_modeling"],
                    "forbidden_objectives": [
                        "entity_supervision",
                        "pseudo_labeling",
                        "synonym_contrastive",
                        "threshold_calibration",
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    terminology = root / terminology_relative_path
    terminology.parent.mkdir(parents=True, exist_ok=True)
    terminology.write_text('{"concept_id":"C1"}\n', encoding="utf-8")
    pairs = root / "derived/pairs.jsonl"
    pairs.write_text(
        '{"concept_id":"C1","left":"ho","right":"ho kéo dài"}\n'
        '{"concept_id":"C2","left":"sốt","right":"sốt cao"}\n',
        encoding="utf-8",
    )
    pairs_manifest = root / "derived/pairs.jsonl.manifest.json"
    pairs_manifest.write_text(
        json.dumps(
            {
                "schema_version": "terminology-synonym-pairs.v1",
                "record_count": 2,
                "dataset_sha256": _sha256(pairs),
                "source_fingerprints": {
                    terminology_relative_path: _sha256(terminology),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = root / "dapt.yaml"
    config.write_text(
        f"""
schema_version: xlmr-dapt-run.v1
run_id: fixture-dapt
run_root: .
environment:
  lock_path: uv.lock
  lock_sha256: {_sha256(lock)}
model:
  model_id: FacebookAI/xlm-roberta-base
  revision: e73636d4f797dec63c3081bb6ed5c7b0bb3f2089
  source_url: https://huggingface.co/FacebookAI/xlm-roberta-base
  license: MIT
corpus:
  manifest: derived/corpus-manifest.json
synonym_pairs:
  path: derived/pairs.jsonl
  manifest: derived/pairs.jsonl.manifest.json
  sources:
    - path: {terminology_relative_path}
      sha256: {_sha256(terminology)}
training:
  output_dir: outputs/model
  cache_dir: .cache/model-training
  max_length: 64
  mlm_batch_size: 2
  contrastive_batch_size: 2
  max_steps: 2
  checkpoint_interval: 1
  preprocessing_workers: 1
runtime:
  operating_system: linux
  accelerator: cuda
  minimum_devices: 1
  minimum_vram_gib: 16
  minimum_compute_capability: [8, 0]
  precision: bf16
""".lstrip(),
        encoding="utf-8",
    )
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
