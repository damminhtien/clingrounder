"""Benchmark plugin discovery and lazy dispatch contracts."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import clingrounder.cli.parser as core_parser
from clingrounder.benchmarks.registry import (
    benchmark_plugins,
    resolve_benchmark_handler,
)
from clingrounder.cli import main
from clingrounder.cli.main import _HANDLERS
from clingrounder.cli.parser import build_parser


def test_phase1_is_an_optional_benchmark_plugin() -> None:
    plugins = benchmark_plugins()

    assert [plugin.name for plugin in plugins] == ["phase1"]
    assert "benchmark_phase1_submission" not in _HANDLERS
    target = resolve_benchmark_handler("benchmark_phase1_submission")
    assert target is not None
    assert target.module == "clingrounder.benchmarks.phase1.commands"
    assert target.function == "run_phase1_submission"
    assert not hasattr(core_parser, "_register_phase1_benchmark_parser")


def test_core_packages_do_not_import_phase1_plugin() -> None:
    """Keep archived task policy outside reusable clinical NLP modules."""

    package_root = Path("src/clingrounder")
    core_packages = (
        "adapters",
        "context",
        "dictionaries",
        "evaluation",
        "kg",
        "linking",
        "mining",
        "ner",
        "ontology",
        "pipeline",
        "preprocessing",
        "relations",
        "retrieval",
        "schema",
        "terminology",
        "training",
        "utils",
        "validation",
    )
    violations: list[tuple[str, str]] = []
    for package in core_packages:
        for path in (package_root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.append(node.module)
                violations.extend(
                    (str(path), module)
                    for module in modules
                    if module.startswith("clingrounder.benchmarks.phase1")
                )

    assert violations == []


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
