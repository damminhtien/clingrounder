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
    _data_parser(commands)
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


def _data_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    data = commands.add_parser("data", help="Acquire, curate, and freeze mined datasets.")
    operations = data.add_subparsers(dest="data_command", required=True)

    registry = operations.add_parser("registry", help="Validate source-registry policy.")
    registry_operations = registry.add_subparsers(
        dest="data_registry_command", required=True
    )
    registry_validate = registry_operations.add_parser(
        "validate", help="Validate registry v2 and print a source summary."
    )
    registry_validate.set_defaults(handler="data_registry_validate")
    registry_validate.add_argument(
        "--registry", default="data/sources/mining_registry.yaml"
    )

    source = operations.add_parser("source", help="Synchronize one registered source.")
    source_operations = source.add_subparsers(dest="data_source_command", required=True)
    source_sync = source_operations.add_parser(
        "sync", help="Discover and checkpoint source artifacts."
    )
    source_sync.set_defaults(handler="data_source_sync")
    source_sync.add_argument("--registry", default="data/sources/mining_registry.yaml")
    source_sync.add_argument("--source-id", required=True)
    source_sync.add_argument("--source-version", required=True)
    source_sync.add_argument("--parameters", help="YAML/JSON request-parameter mapping.")
    source_sync.add_argument("--store", required=True)
    source_sync.add_argument("--encrypted-at-rest", action="store_true")
    source_sync.add_argument("--output", required=True)

    dataset = operations.add_parser("dataset", help="Build documents from acquired artifacts.")
    dataset_operations = dataset.add_subparsers(
        dest="data_dataset_command", required=True
    )
    dataset_build = dataset_operations.add_parser(
        "build", help="Parse a source artifact manifest into immutable documents."
    )
    dataset_build.set_defaults(handler="data_dataset_build")
    dataset_build.add_argument("--registry", default="data/sources/mining_registry.yaml")
    dataset_build.add_argument("--source-id", required=True)
    dataset_build.add_argument("--artifacts", required=True)
    dataset_build.add_argument("--store", required=True)
    dataset_build.add_argument("--output", required=True)

    label = operations.add_parser("label", help="Run a local proposal-labeler adapter.")
    label_operations = label.add_subparsers(dest="data_label_command", required=True)
    label_propose = label_operations.add_parser(
        "propose", help="Generate provenance-bearing annotation proposals."
    )
    label_propose.set_defaults(handler="data_label_propose")
    label_propose.add_argument("--documents", required=True)
    label_propose.add_argument(
        "--adapter", required=True, help="Local factory in module:attribute form."
    )
    label_propose.add_argument("--adapter-config", help="YAML/JSON factory config mapping.")
    label_propose.add_argument("--output", required=True)
    label_propose.add_argument("--batch-size", type=int, default=16)

    review = operations.add_parser("review", help="Exchange deterministic review queues.")
    review_operations = review.add_subparsers(dest="data_review_command", required=True)
    review_export = review_operations.add_parser("export", help="Export a JSONL review queue.")
    review_export.set_defaults(handler="data_review_export")
    review_export.add_argument("--documents", required=True)
    review_export.add_argument("--proposals", required=True)
    review_export.add_argument("--output", required=True)
    review_import = review_operations.add_parser(
        "import", help="Import reviewed proposal decisions."
    )
    review_import.set_defaults(handler="data_review_import")
    review_import.add_argument("--input", required=True)
    review_import.add_argument("--output", required=True)

    coverage = operations.add_parser("coverage", help="Measure coverage and review priority.")
    coverage_operations = coverage.add_subparsers(
        dest="data_coverage_command", required=True
    )
    coverage_report = coverage_operations.add_parser(
        "report", help="Write coverage-cube cells and ranked review records."
    )
    coverage_report.set_defaults(handler="data_coverage_report")
    coverage_report.add_argument("--documents", required=True)
    coverage_report.add_argument("--proposals", required=True)
    coverage_report.add_argument("--targets", required=True)
    coverage_report.add_argument("--snapshot-id", required=True)
    coverage_report.add_argument("--output", required=True)

    snapshot = operations.add_parser("snapshot", help="Freeze a leakage-safe snapshot.")
    snapshot_operations = snapshot.add_subparsers(
        dest="data_snapshot_command", required=True
    )
    snapshot_freeze = snapshot_operations.add_parser(
        "freeze", help="Validate and atomically freeze Parquet snapshot shards."
    )
    snapshot_freeze.set_defaults(handler="data_snapshot_freeze")
    snapshot_freeze.add_argument("--documents", required=True)
    snapshot_freeze.add_argument("--annotations")
    snapshot_freeze.add_argument("--relations")
    snapshot_freeze.add_argument("--artifacts")
    snapshot_freeze.add_argument("--version", required=True)
    snapshot_freeze.add_argument("--created-at", required=True)
    snapshot_freeze.add_argument("--output-dir", required=True)
    snapshot_freeze.add_argument("--development-fraction", type=float, default=0.1)
    snapshot_freeze.add_argument("--challenge-source", action="append", default=[])
    snapshot_freeze.add_argument("--challenge-template", action="append", default=[])
    snapshot_freeze.add_argument("--hash-salt", default="medical-kg-snapshot-v1")
    snapshot_freeze.add_argument("--max-synthetic-fraction", type=float, default=0.4)
    snapshot_freeze.add_argument("--manifest-only", action="store_true")

    run = operations.add_parser("run", help="Run a resumable declarative mining plan.")
    run.set_defaults(handler="data_run")
    run.add_argument("--plan", required=True)
