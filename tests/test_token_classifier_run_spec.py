"""Contracts for immutable Linux/CUDA token-classifier run specs."""

from __future__ import annotations

from pathlib import Path

import pytest

from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.training.run_spec import (
    assert_local_gpu_runtime,
    load_token_classifier_run_spec,
)


def test_checked_in_full_type_run_spec_pins_dataset_and_checkpoint() -> None:
    spec = load_token_classifier_run_spec(
        "configs/models/open-corpus-full-type-xlmr-base-2026-07-19.yaml"
    )

    assert spec.training.model_id == "FacebookAI/xlm-roberta-base"
    assert spec.training.revision == "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    assert spec.training.evaluation_split == "development"
    assert spec.runtime.minimum_compute_capability == (8, 0)
    assert spec.runtime.precision == "bf16"


def test_run_spec_rejects_mutable_model_revision(tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text(
        _yaml(revision="main"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        load_token_classifier_run_spec(config)


def test_gpu_gate_fails_clearly_on_non_linux_host(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_token_classifier_run_spec(
        "configs/models/open-corpus-full-type-xlmr-base-2026-07-19.yaml"
    )
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    with pytest.raises(RuntimeError, match="requires linux, current host is darwin"):
        assert_local_gpu_runtime(spec.runtime)


def test_run_spec_cli_is_discoverable() -> None:
    args = build_parser().parse_args(
        [
            "model",
            "train-token-classifier-run",
            "--config",
            "configs/models/run.yaml",
        ]
    )

    assert args.handler == "model_train_token_classifier_run"


def _yaml(*, revision: str) -> str:
    return f"""\
schema_version: token-classifier-run.v1
run_id: fixture
dataset:
  path: spans.jsonl
  manifest: manifest.json
  train_split: train
  evaluation_split: development
model:
  model_id: local/model
  revision: {revision}
  source_url: https://example.test/model
  license: MIT
training:
  output_dir: outputs/model
runtime:
  operating_system: linux
  accelerator: cuda
  minimum_devices: 1
  minimum_vram_gib: 16
  minimum_compute_capability: [8, 0]
  precision: bf16
"""
