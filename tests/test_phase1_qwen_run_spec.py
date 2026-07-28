"""Pinned model, parameter-budget, and cost contracts for Qwen runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.qwen_runner import _build_qwen_runtime
from medical_kg_nlp.benchmarks.phase1.qwen_run_spec import (
    load_phase1_qwen_run_spec,
)
from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.utils.hashing import (
    sha256_directory,
    sha256_file,
)


@pytest.mark.parametrize(
    ("config_name", "parameter_count"),
    [
        ("phase1-qwen3-8b-2026-07-27.yaml", 8_190_735_360),
        ("phase1-qwen2.5-7b-control-2026-07-27.yaml", 7_615_616_512),
    ],
)
def test_checked_in_qwen_run_specs_are_pinned(
    config_name: str,
    parameter_count: int,
) -> None:
    spec = load_phase1_qwen_run_spec(Path("configs/models") / config_name)

    assert spec.model.parameter_count == parameter_count
    assert spec.budget.total_parameters == parameter_count
    assert spec.maximum_vast_cost_usd == 6.0
    assert spec.prefetch_command[:2] == ("hf", "download")
    assert spec.dataset_path.name == "extraction.jsonl"
    assert "adapter" not in spec.to_dict()


def test_qwen_run_spec_rejects_cost_above_user_limit(tmp_path: Path) -> None:
    source = Path("configs/models/phase1-qwen3-8b-2026-07-27.yaml").read_text(
        encoding="utf-8"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        source.replace("run_root: ../..", "run_root: .").replace(
            "maximum_vast_cost_usd: 6.0",
            "maximum_vast_cost_usd: 6.01",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot exceed USD 6"):
        load_phase1_qwen_run_spec(config)


def test_qwen_run_spec_pins_local_peft_provenance_and_budget(
    tmp_path: Path,
) -> None:
    config = _write_peft_qwen_spec(tmp_path, adapter_parameter_count=12_000_000)

    spec = load_phase1_qwen_run_spec(config)
    verification = spec.verify_adapter_inputs()
    runtime = _build_qwen_runtime(spec)

    assert spec.adapter is not None
    assert spec.budget.total_parameters == 8_202_735_360
    assert spec.budget.total_parameters <= 9_000_000_000
    assert verification is not None
    assert verification["fingerprint"] == spec.adapter.fingerprint
    assert runtime.adapter is not None
    assert runtime.adapter.path == spec.adapter.path
    assert runtime.adapter.parameter_count == 12_000_000
    serialized = spec.to_dict()
    assert serialized["adapter"]["provenance"]["manifest_sha256"] == (
        spec.adapter.provenance_manifest_sha256
    )


def test_qwen_run_spec_rejects_changed_peft_bytes(tmp_path: Path) -> None:
    config = _write_peft_qwen_spec(tmp_path, adapter_parameter_count=12_000_000)
    spec = load_phase1_qwen_run_spec(config)
    assert spec.adapter is not None
    (spec.adapter.path / "adapter_model.safetensors").write_bytes(b"changed")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        spec.verify_adapter_inputs()


def test_qwen_run_spec_counts_adapter_against_nine_billion_limit(
    tmp_path: Path,
) -> None:
    config = _write_peft_qwen_spec(tmp_path, adapter_parameter_count=900_000_000)

    with pytest.raises(ValueError, match="budget exceeded"):
        load_phase1_qwen_run_spec(config)


def test_qwen_cli_commands_are_discoverable() -> None:
    parser = build_parser()

    inspect_args = parser.parse_args(
        [
            "benchmark",
            "phase1",
            "qwen",
            "inspect",
            "--config",
            "qwen.yaml",
        ]
    )
    data_args = parser.parse_args(
        ["benchmark", "phase1", "model-data", "build-qwen"]
    )
    propose_args = parser.parse_args(
        [
            "benchmark",
            "phase1",
            "qwen",
            "propose",
            "--config",
            "qwen.yaml",
            "--documents",
            "documents.jsonl",
            "--source-archive-sha256",
            "a" * 64,
            "--output-dir",
            "output",
        ]
    )

    assert inspect_args.handler == "benchmark_phase1_qwen_inspect"
    assert data_args.handler == "benchmark_phase1_qwen_data_build"
    assert propose_args.handler == "benchmark_phase1_qwen_propose"


def _write_peft_qwen_spec(
    root: Path,
    *,
    adapter_parameter_count: int,
) -> Path:
    adapter_dir = root / "outputs/models/qwen-adapter/final-adapter"
    adapter_dir.mkdir(parents=True)
    adapter_config = {
        "base_model_name_or_path": "Qwen/Qwen3-8B",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config_path.write_text(
        json.dumps(adapter_config, sort_keys=True),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    training_commit = "2" * 40
    manifest = {
        "schema_version": "causal-qlora-artifact.v1",
        "model": {
            "model_id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "parameter_count": 8_190_735_360,
        },
        "source_control": {
            "git_commit": training_commit,
            "git_dirty": False,
            "working_tree_hash": None,
        },
        "artifacts": {
            "adapter_config_sha256": sha256_file(adapter_config_path),
        },
    }
    manifest_path = adapter_dir.parent / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    source = Path("configs/models/phase1-qwen3-8b-2026-07-27.yaml").read_text(
        encoding="utf-8"
    )
    config = root / "qwen-peft.yaml"
    config.write_text(
        source.replace("run_root: ../..", "run_root: .")
        + f"""

adapter:
  artifact_id: qwen3-phase1-qlora
  model_id: local/qwen3-phase1-qlora
  revision: "{training_commit}"
  parameter_count: {adapter_parameter_count}
  kind: adapter
  roles: [adjudication, recall, targeted]
  path: outputs/models/qwen-adapter/final-adapter
  fingerprint: {sha256_directory(adapter_dir)}
  provenance:
    manifest: outputs/models/qwen-adapter/run_manifest.json
    manifest_sha256: {sha256_file(manifest_path)}
""",
        encoding="utf-8",
    )
    return config
