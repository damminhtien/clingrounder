"""Contracts for immutable Linux/CUDA token-classifier run specs."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.cli.commands import model as model_commands
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
    assert spec.training.full_determinism is True
    assert spec.config_relative_path == (
        "configs/models/open-corpus-full-type-xlmr-base-2026-07-19.yaml"
    )
    assert spec.training.dataset_path.is_absolute()


def test_run_spec_paths_are_stable_from_another_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "portable-repository"
    config = repository / "configs" / "models" / "run.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(_yaml(revision="a" * 40, run_root="../.."), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    spec = load_token_classifier_run_spec(config)
    serialized = spec.training.to_dict(path_root=spec.run_root)

    assert spec.run_root == repository.resolve()
    assert spec.training.dataset_path == (repository / "spans.jsonl").resolve()
    assert serialized["dataset_path"] == "spans.jsonl"
    assert serialized["output_dir"] == "outputs/model"
    assert str(repository) not in str(serialized)


def test_run_spec_rejects_paths_outside_declared_root(tmp_path: Path) -> None:
    config = tmp_path / "repository" / "run.yaml"
    config.parent.mkdir()
    config.write_text(
        _yaml(revision="a" * 40).replace("path: spans.jsonl", "path: ../spans.jsonl"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset.path escapes declared run_root"):
        load_token_classifier_run_spec(config)


def test_run_cli_persists_only_run_root_relative_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "portable-repository"
    config = repository / "configs" / "models" / "run.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(_yaml(revision="a" * 40, run_root="../.."), encoding="utf-8")
    observed: dict[str, Path] = {}

    def fake_train(
        _training: object,
        *,
        manifest_root: Path | None = None,
    ) -> dict[str, object]:
        assert manifest_root is not None
        observed["manifest_root"] = manifest_root
        return {
            "schema_version": "token-classifier-training.v2",
            "model": {"output": "outputs/model/final-model"},
            "metrics": {},
        }

    monkeypatch.setattr(model_commands, "assert_local_gpu_runtime", lambda _: {})
    monkeypatch.setattr(
        model_commands,
        "train_huggingface_token_classifier",
        fake_train,
    )

    assert model_commands.train_token_classifier_run(Namespace(config=str(config))) == 0
    manifest_path = repository / "outputs" / "model" / "run_manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert observed["manifest_root"] == repository.resolve()
    assert manifest["run_spec"]["path"] == "configs/models/run.yaml"
    assert manifest["run_spec"]["path_base"] == "run_root"
    assert str(repository) not in manifest_text
    assert json.loads(capsys.readouterr().out)["manifest"] == ("outputs/model/run_manifest.json")


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


def _yaml(*, revision: str, run_root: str = ".") -> str:
    return f"""\
schema_version: token-classifier-run.v2
run_id: fixture
run_root: {run_root}
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
  full_determinism: true
runtime:
  operating_system: linux
  accelerator: cuda
  minimum_devices: 1
  minimum_vram_gib: 16
  minimum_compute_capability: [8, 0]
  precision: bf16
"""
