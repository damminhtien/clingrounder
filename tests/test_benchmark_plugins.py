"""Benchmark plugin discovery and lazy dispatch contracts."""

from __future__ import annotations

import json

import medical_kg_nlp.cli.parser as core_parser
from medical_kg_nlp.benchmarks.registry import (
    benchmark_plugins,
    resolve_benchmark_handler,
)
from medical_kg_nlp.cli import main
from medical_kg_nlp.cli.main import _HANDLERS
from medical_kg_nlp.cli.parser import build_parser


def test_phase1_is_an_optional_benchmark_plugin() -> None:
    plugins = benchmark_plugins()

    assert [plugin.name for plugin in plugins] == ["phase1"]
    assert "benchmark_phase1_submission" not in _HANDLERS
    target = resolve_benchmark_handler("benchmark_phase1_submission")
    assert target is not None
    assert target.module == "medical_kg_nlp.benchmarks.phase1.commands"
    assert target.function == "run_phase1_submission"
    assert not hasattr(core_parser, "_register_phase1_benchmark_parser")


def test_benchmark_list_is_task_neutral(capsys) -> None:
    assert main(["benchmark", "list"]) == 0

    assert json.loads(capsys.readouterr().out) == [
        {
            "name": "phase1",
            "summary": "Archived Vietnamese medical extraction challenge benchmark",
        }
    ]


def test_phase1_plugin_preserves_existing_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "submission",
            "--input-dir",
            "input",
            "--assertion-overlay",
            "private/assertions.jsonl",
        ]
    )

    assert args.handler == "benchmark_phase1_submission"
    assert args.assertion_overlay == "private/assertions.jsonl"
