"""Portable run-spec and CLI contracts for Qwen QLoRA stages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.training.causal_artifact import finalize_causal_qlora_artifact
from medical_kg_nlp.training.causal_qlora import inspect_causal_qlora_inputs
from medical_kg_nlp.training.causal_run_spec import load_causal_qlora_run_spec
from medical_kg_nlp.utils.hashing import sha256_file


def test_checked_in_qwen_stages_pin_model_data_and_adapter_handoff() -> None:
    curriculum = load_causal_qlora_run_spec(
        "configs/models/phase1-qwen3-8b-qlora-curriculum-2026-07-28.yaml"
    )
    specialize = load_causal_qlora_run_spec(
        "configs/models/phase1-qwen3-8b-qlora-specialize-2026-07-28.yaml"
    )

    assert curriculum.training.parameter_count == 8_190_735_360
    assert curriculum.training.revision == (
        "b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert curriculum.training.max_length == 2048
    assert curriculum.maximum_vast_cost_usd == 6.0
    assert specialize.training.initial_adapter_path == (
        curriculum.training.output_dir / "final-adapter"
    )
    assert specialize.training.max_length == 4096
    assert specialize.training.evaluation_sources[0].split == "development"
    assert len(curriculum.training.train_sources) == 2
    assert curriculum.training.train_sources[0].maximum_records == 800
    assert curriculum.training.train_sources[0].sha256 == (
        "cfe2274d07621d63698c9d667c15ad6a509f202a6c988c92cbed011159dfa09f"
    )


def test_inspect_reports_portable_fixture_counts(tmp_path: Path) -> None:
    run_spec = load_causal_qlora_run_spec(_write_run_spec(tmp_path))

    report = inspect_causal_qlora_inputs(run_spec.training)

    assert report["train_record_count"] == 1
    assert report["evaluation_record_count"] == 0


def test_run_spec_rejects_changed_lockfile(tmp_path: Path) -> None:
    config = _write_run_spec(tmp_path)
    (tmp_path / "uv.lock").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lock SHA-256 mismatch"):
        load_causal_qlora_run_spec(config)


def test_run_spec_rejects_model_over_parameter_budget(tmp_path: Path) -> None:
    config = _write_run_spec(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "parameter_count: 8190735360",
            "parameter_count: 9000000001",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="9B parameter limit"):
        load_causal_qlora_run_spec(config)


def test_qlora_cli_commands_are_discoverable() -> None:
    parser = build_parser()

    inspect_args = parser.parse_args(
        [
            "model",
            "inspect-causal-qlora-run",
            "--config",
            "run.yaml",
        ]
    )
    finalize_args = parser.parse_args(
        [
            "model",
            "finalize-causal-qlora-run",
            "--config",
            "run.yaml",
        ]
    )
    train_args = parser.parse_args(
        [
            "model",
            "train-causal-qlora-run",
            "--config",
            "run.yaml",
            "--max-steps",
            "1",
            "--output-dir",
            "outputs/smoke",
        ]
    )

    assert inspect_args.handler == "model_inspect_causal_qlora_run"
    assert finalize_args.handler == "model_finalize_causal_qlora_run"
    assert train_args.handler == "model_train_causal_qlora_run"
    assert train_args.max_steps == 1


def test_finalize_qlora_artifact_uses_portable_source_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_run_spec(tmp_path)
    spec = load_causal_qlora_run_spec(config_path)
    output_dir = spec.training.output_dir
    adapter_dir = output_dir / "final-adapter"
    adapter_dir.mkdir(parents=True)
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_config.write_text(
        '{"base_model_name_or_path":"Qwen/Qwen3-8B"}\n',
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "causal-qlora-artifact.v1",
        "model": {
            "model_id": spec.training.model_id,
            "revision": spec.training.revision,
            "parameter_count": spec.training.parameter_count,
        },
        "run_spec": {"sha256": sha256_file(config_path)},
        "environment": {
            "lock_sha256": spec.environment_lock_sha256,
        },
        "artifacts": {
            "adapter_config_sha256": sha256_file(adapter_config),
        },
        "source_control": {
            "git_commit": None,
            "git_dirty": None,
            "working_tree_hash": None,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    source_commit = "b" * 40
    (tmp_path / ".source-commit").write_text(
        source_commit + "\n",
        encoding="ascii",
    )
    monkeypatch.chdir(tmp_path)

    report = finalize_causal_qlora_artifact(
        output_dir,
        model_id=spec.training.model_id,
        model_revision=spec.training.revision,
        parameter_count=spec.training.parameter_count,
        run_spec_path=config_path,
        run_spec_sha256=sha256_file(config_path),
        environment_lock_path=spec.environment_lock_path,
        environment_lock_sha256=spec.environment_lock_sha256,
    )

    finalized = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "finalized"
    assert finalized["source_control"]["git_commit"] == source_commit
    assert (
        finalized["source_control"]["source_control_mode"]
        == "source_commit_marker"
    )


def _write_run_spec(root: Path) -> Path:
    lock = root / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    dataset = root / "data.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "record_id": "one",
                "document_id": "doc-one",
                "split": "train",
                "messages": [
                    {"role": "user", "content": "source"},
                    {"role": "assistant", "content": "{}"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = root / "run.yaml"
    config.write_text(
        f"""
schema_version: causal-qlora-run.v1
run_id: fixture
run_root: .
model:
  model_id: Qwen/Qwen3-8B
  revision: {"a" * 40}
  parameter_count: 8190735360
  source_url: https://example.invalid/model
  license: Apache-2.0
datasets:
  train:
    - path: data.jsonl
      sha256: {sha256_file(dataset)}
      split: train
  evaluation: []
training:
  output_dir: outputs/model
  max_steps: -1
runtime:
  minimum_compute_capability: [8, 0]
environment:
  lock_path: uv.lock
  lock_sha256: {sha256_file(lock)}
remote:
  maximum_vast_cost_usd: 6
""".lstrip(),
        encoding="utf-8",
    )
    return config
