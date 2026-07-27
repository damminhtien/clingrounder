"""Pinned model, parameter-budget, and cost contracts for Qwen runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.qwen_run_spec import (
    load_phase1_qwen_run_spec,
)
from medical_kg_nlp.cli.parser import build_parser


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

    assert inspect_args.handler == "benchmark_phase1_qwen_inspect"
    assert data_args.handler == "benchmark_phase1_qwen_data_build"
