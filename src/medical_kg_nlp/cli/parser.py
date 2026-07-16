"""Argument parser definitions without command implementation imports."""

from __future__ import annotations

import argparse

from medical_kg_nlp.validation.profiles import ValidationProfile

__all__ = ["build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Build the stable `medical-kg` command hierarchy."""

    parser = argparse.ArgumentParser(prog="medical-kg", description="Medical KG NLP tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    _pipeline_parser(commands)
    _terminology_parser(commands)
    _evaluate_parser(commands)
    _validate_parser(commands)
    _benchmark_parser(commands)
    return parser


def _pipeline_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    pipeline = commands.add_parser("pipeline", help="Run pipeline operations.")
    operations = pipeline.add_subparsers(dest="pipeline_command", required=True)
    run = operations.add_parser("run", help="Run the configured pipeline over JSONL documents.")
    run.set_defaults(handler="pipeline_run")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--config")
    run.add_argument("--dictionary")
    run.add_argument("--abbreviations")
    run.add_argument("--run-root")
    run.add_argument("--run-label", default="pipeline")
    run.add_argument("--parallel-backend", choices=("serial", "thread", "process"), default="process")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--chunksize", type=int, default=4)
    run.add_argument("--no-fail-fast", action="store_true")


def _terminology_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    terminology = commands.add_parser("terminology", help="Build or inspect terminology indexes.")
    operations = terminology.add_subparsers(dest="terminology_command", required=True)
    build = operations.add_parser("build", help="Build a versioned SQLite FTS5 index.")
    build.set_defaults(handler="terminology_build")
    build.add_argument("--source", action="append", required=True)
    build.add_argument("--output")
    build.add_argument("--cache-dir", default=".cache/medical-kg/terminology")

    inspect = operations.add_parser("inspect", help="Inspect metadata or query a built index.")
    inspect.set_defaults(handler="terminology_inspect")
    inspect.add_argument("--index", required=True)
    inspect.add_argument("--source", action="append")
    inspect.add_argument("--query")
    inspect.add_argument("--entity-type")
    inspect.add_argument("--code-system", action="append")
    inspect.add_argument("--limit", type=int, default=20)


def _evaluate_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    evaluate = commands.add_parser("evaluate", help="Evaluate internal-schema predictions.")
    evaluate.set_defaults(handler="evaluate")
    evaluate.add_argument("--gold", required=True)
    evaluate.add_argument("--pred", required=True)
    evaluate.add_argument("--error-analysis", default="outputs/error_analysis.csv")


def _validate_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    validate = commands.add_parser("validate", help="Validate predictions and optional artifacts.")
    validate.set_defaults(handler="validate")
    validate.add_argument("--pred", required=True)
    validate.add_argument("--documents")
    validate.add_argument("--dictionary")
    validate.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in ValidationProfile),
        default=ValidationProfile.DEVELOPMENT.value,
    )
    validate.add_argument("--artifact")
    validate.add_argument("--expected-file", action="append", default=[])


def _benchmark_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    benchmark = commands.add_parser("benchmark", help="Run task-specific benchmark plugins.")
    plugins = benchmark.add_subparsers(dest="benchmark_name", required=True)
    phase1 = plugins.add_parser("phase1", help="Build a strict Phase 1 submission artifact.")
    phase1.set_defaults(handler="benchmark_phase1")
    phase1.add_argument("--input-dir", required=True)
    phase1.add_argument("--output-dir", required=True)
    phase1.add_argument("--zip", required=True)
    phase1.add_argument("--dictionary", default="data/dictionaries/seed_concepts.jsonl")
    phase1.add_argument("--abbreviations", default="data/dictionaries/abbreviations.jsonl")
    phase1.add_argument("--assertion-policy", choices=("empty", "pipeline"), default="pipeline")
    phase1.add_argument("--candidate-policy", choices=("empty", "pipeline"), default="pipeline")
    phase1.add_argument("--max-candidates", type=int, default=5)
    phase1.add_argument("--parallel-backend", choices=("serial", "thread", "process"), default="process")
    phase1.add_argument("--workers", type=int, default=1)
    phase1.add_argument("--chunksize", type=int, default=4)
