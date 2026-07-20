"""Argument parser definitions without command implementation imports."""

from __future__ import annotations

import argparse

from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.validation.profiles import ValidationProfile

__all__ = ["build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Build the stable `medical-kg` command hierarchy."""

    parser = argparse.ArgumentParser(prog="medical-kg", description="Medical KG NLP tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    _pipeline_parser(commands)
    _terminology_parser(commands)
    _kg_parser(commands)
    _evaluate_parser(commands)
    _validate_parser(commands)
    _benchmark_parser(commands)
    _model_parser(commands)
    _data_parser(commands)
    return parser


def _kg_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    kg = commands.add_parser("kg", help="Build or query persistent knowledge graphs.")
    operations = kg.add_subparsers(dest="kg_command", required=True)
    build = operations.add_parser("build", help="Build a versioned SQLite graph index.")
    build.set_defaults(handler="kg_build")
    build.add_argument("--nodes", required=True)
    build.add_argument("--edges", required=True)
    build.add_argument("--evidence", required=True)
    build.add_argument("--output")
    build.add_argument("--manifest-output")
    build.add_argument("--cache-dir", default=".cache/medical-kg/knowledge-graph")

    inspect = operations.add_parser("inspect", help="Search or traverse a graph index.")
    inspect.set_defaults(handler="kg_inspect")
    inspect.add_argument("--index", required=True)
    inspect.add_argument("--nodes")
    inspect.add_argument("--edges")
    inspect.add_argument("--evidence")
    inspect.add_argument("--query")
    inspect.add_argument("--entity-type")
    inspect.add_argument("--code-system")
    inspect.add_argument("--code")
    inspect.add_argument("--node-id")
    inspect.add_argument("--edge-id")
    inspect.add_argument("--relation-type", action="append", default=[])
    inspect.add_argument(
        "--direction",
        choices=("outgoing", "incoming", "both"),
        default="outgoing",
    )
    inspect.add_argument("--min-support", type=int, default=1)
    inspect.add_argument("--ancestors", action="store_true")
    inspect.add_argument("--max-depth", type=int, default=20)
    inspect.add_argument("--limit", type=int, default=20)

    benchmark = operations.add_parser(
        "benchmark-aliases",
        help="Benchmark exact alias coverage against overlay targets.",
    )
    benchmark.set_defaults(handler="kg_benchmark_aliases")
    benchmark.add_argument("--index", required=True)
    benchmark.add_argument("--alias-overlay", action="append", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--limit", type=int, default=5)
    benchmark.add_argument("--max-misses", type=int, default=50)

    relation_benchmark = operations.add_parser(
        "benchmark-relations",
        help="Benchmark relation-edge coverage, latency, and concurrent graph reads.",
    )
    relation_benchmark.set_defaults(handler="kg_benchmark_relations")
    relation_benchmark.add_argument("--index", required=True)
    relation_benchmark.add_argument("--edges", required=True)
    relation_benchmark.add_argument("--relation-type", required=True)
    relation_benchmark.add_argument("--output", required=True)
    relation_benchmark.add_argument("--workers", type=int, default=8)
    relation_benchmark.add_argument("--repeats", type=int, default=3)
    relation_benchmark.add_argument("--limit", type=int, default=100)
    relation_benchmark.add_argument("--max-misses", type=int, default=50)

    reranker_benchmark = operations.add_parser(
        "benchmark-reranker",
        help="Calibrate and evaluate graph evidence over terminology candidates.",
    )
    reranker_benchmark.set_defaults(handler="kg_benchmark_reranker")
    reranker_benchmark.add_argument("--index", required=True)
    reranker_benchmark.add_argument("--nodes", required=True)
    reranker_benchmark.add_argument("--edges", required=True)
    reranker_benchmark.add_argument("--evidence", required=True)
    reranker_benchmark.add_argument("--terminology-index", required=True)
    reranker_benchmark.add_argument("--terminology-source", action="append", required=True)
    reranker_benchmark.add_argument("--terminology-alias-overlay", action="append", default=[])
    reranker_benchmark.add_argument("--documents", required=True)
    reranker_benchmark.add_argument("--annotations", required=True)
    reranker_benchmark.add_argument(
        "--predictions",
        help=(
            "Pipeline prediction JSONL required by predicted_ner_exact_unique mode."
        ),
    )
    reranker_benchmark.add_argument("--output", required=True)
    reranker_benchmark.add_argument("--calibration-split", default="dev")
    reranker_benchmark.add_argument("--evaluation-split", default="test")
    reranker_benchmark.add_argument("--graph-source-split", action="append", default=["train"])
    reranker_benchmark.add_argument("--document-prefix", default="codiesp:")
    reranker_benchmark.add_argument("--source-label", default="DIAGNOSTICO")
    reranker_benchmark.add_argument(
        "--context-mode",
        choices=("oracle", "predicted_exact_unique", "predicted_ner_exact_unique"),
        default="oracle",
    )
    reranker_benchmark.add_argument(
        "--relation-type",
        action="append",
        default=["CO_OCCURS_WITH"],
    )
    reranker_benchmark.add_argument("--min-support", type=int, default=2)
    reranker_benchmark.add_argument("--candidate-limit", type=int, default=20)
    reranker_benchmark.add_argument(
        "--max-bonus-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.02, 0.04, 0.08],
    )
    reranker_benchmark.add_argument("--max-errors", type=int, default=50)


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
    build.add_argument(
        "--alias-overlay",
        action="append",
        default=[],
        help="Strict provenance-bearing alias JSONL; may be repeated.",
    )
    build.add_argument("--output")
    build.add_argument(
        "--manifest-output",
        help="Persist the reproducibility manifest in addition to printing it.",
    )
    build.add_argument("--cache-dir", default=".cache/medical-kg/terminology")

    query_set = operations.add_parser(
        "query-set",
        help="Build neutral retrieval queries from overlays or held-out linked proposals.",
    )
    query_set.set_defaults(handler="terminology_query_set")
    query_sources = query_set.add_mutually_exclusive_group(required=True)
    query_sources.add_argument("--alias-overlay", action="append")
    query_sources.add_argument("--linked-proposal", action="append")
    query_set.add_argument(
        "--reference-alias-overlay",
        action="append",
        default=[],
        help="Training overlays used only to label held-out seen/unseen query slices.",
    )
    query_set.add_argument("--output", required=True)
    query_set.add_argument("--manifest-output", required=True)

    inspect = operations.add_parser("inspect", help="Inspect metadata or query a built index.")
    inspect.set_defaults(handler="terminology_inspect")
    inspect.add_argument("--index", required=True)
    inspect.add_argument("--source", action="append")
    inspect.add_argument("--alias-overlay", action="append", default=[])
    inspect.add_argument("--query")
    inspect.add_argument("--entity-type")
    inspect.add_argument("--code-system", action="append")
    inspect.add_argument("--limit", type=int, default=20)

    benchmark = operations.add_parser(
        "benchmark",
        help="Evaluate exact, toneless, and FTS retrieval against neutral queries.",
    )
    benchmark.set_defaults(handler="terminology_benchmark")
    benchmark.add_argument("--index", required=True)
    benchmark.add_argument("--source", action="append", required=True)
    benchmark.add_argument("--alias-overlay", action="append", default=[])
    benchmark.add_argument("--queries", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--limit", type=int, default=20)
    benchmark.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed query errors; they are always retained in --output.",
    )


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
    phase1.add_argument(
        "--pipeline-config",
        help=(
            "Optional reusable pipeline profile. Its mined recognition sources and full "
            "terminology index are preserved while benchmark paths override the base files."
        ),
    )
    phase1.add_argument(
        "--validation-dictionary",
        dest="validation_dictionaries",
        action="append",
        default=[],
        help=(
            "Canonical JSONL source accepted by release validation; repeat for TT06 and "
            "RxNorm when a full terminology profile emits both systems."
        ),
    )
    phase1.add_argument("--assertion-policy", choices=("empty", "pipeline"), default="pipeline")
    phase1.add_argument("--candidate-policy", choices=("empty", "pipeline"), default="pipeline")
    phase1.add_argument("--max-candidates", type=int, default=5)
    phase1.add_argument("--parallel-backend", choices=("serial", "thread", "process"), default="process")
    phase1.add_argument("--workers", type=int, default=1)
    phase1.add_argument("--chunksize", type=int, default=4)


def _model_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    model = commands.add_parser("model", help="Validate datasets and train local models.")
    operations = model.add_subparsers(dest="model_command", required=True)

    validate = operations.add_parser(
        "validate-token-dataset",
        help="Validate mined spans and train/evaluation label compatibility.",
    )
    validate.set_defaults(handler="model_validate_token_dataset")
    _token_training_identity_arguments(validate, include_output=False, include_model=False)

    train = operations.add_parser(
        "train-token-classifier",
        help="Train a pinned, locally cached Hugging Face token classifier.",
    )
    train.set_defaults(handler="model_train_token_classifier")
    _token_training_identity_arguments(train, include_output=True, include_model=True)
    train.add_argument("--max-length", type=int, default=512)
    train.add_argument("--stride", type=int, default=64)
    train.add_argument("--train-batch-size", type=int, default=8)
    train.add_argument("--evaluation-batch-size", type=int, default=16)
    train.add_argument("--epochs", type=float, default=3.0)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-ratio", type=float, default=0.1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--preprocessing-workers", type=int, default=1)
    train.add_argument("--seed", type=int, default=42)
    precision = train.add_mutually_exclusive_group()
    precision.add_argument("--fp16", action="store_true")
    precision.add_argument("--bf16", action="store_true")
    train.add_argument("--cpu", action="store_true")
    train.add_argument("--cache-dir")
    train.add_argument("--resume-from-checkpoint")
    train.add_argument("--overwrite-output", action="store_true")

    inspect_run = operations.add_parser(
        "inspect-token-classifier-run",
        help="Validate a pinned token-classifier run spec without ML imports.",
    )
    inspect_run.set_defaults(handler="model_inspect_token_classifier_run")
    inspect_run.add_argument("--config", required=True)

    train_run = operations.add_parser(
        "train-token-classifier-run",
        help="Validate Linux/CUDA and execute one pinned token-classifier run spec.",
    )
    train_run.set_defaults(handler="model_train_token_classifier_run")
    train_run.add_argument("--config", required=True)


def _token_training_identity_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_output: bool,
    include_model: bool,
) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    if include_model:
        parser.add_argument("--model-id", required=True)
        parser.add_argument("--revision", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--evaluation-split", default="development")
    parser.add_argument(
        "--internal-validation-fraction",
        type=float,
        default=0.0,
        help=(
            "Deterministically hold out train documents for model selection; "
            "mutually exclusive with --evaluation-split."
        ),
    )
    parser.add_argument(
        "--no-evaluation",
        action="store_true",
        help="Train without an evaluation split; intended only for final refits.",
    )
    if include_output:
        parser.add_argument("--output-dir", required=True)


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
    registry_validate.add_argument(
        "--processing-index",
        help="Optional source-processing status file whose docs/config paths must exist.",
    )
    registry_validate.add_argument(
        "--repository-root",
        default=".",
        help="Repository root used to validate processing-index paths.",
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
    dataset_inspect = dataset_operations.add_parser(
        "inspect", help="Profile documents, labels, duplicates, and offset integrity."
    )
    dataset_inspect.set_defaults(handler="data_dataset_inspect")
    dataset_inspect.add_argument("--documents", required=True)
    dataset_inspect.add_argument("--annotations")
    dataset_inspect.add_argument("--output", required=True)
    dataset_inspect.add_argument(
        "--strict",
        action="store_true",
        help="Return a failing status when structural issues are present.",
    )
    dataset_reconcile = dataset_operations.add_parser(
        "reconcile-duplicates",
        help="Collapse exact-text duplicates and separate consensus from disagreements.",
    )
    dataset_reconcile.set_defaults(handler="data_dataset_reconcile_duplicates")
    dataset_reconcile.add_argument("--documents", required=True)
    dataset_reconcile.add_argument("--annotations", required=True)
    dataset_reconcile.add_argument("--documents-output", required=True)
    dataset_reconcile.add_argument("--annotations-output", required=True)
    dataset_reconcile.add_argument("--review-output", required=True)
    dataset_reconcile.add_argument("--mapping-output", required=True)
    dataset_reconcile.add_argument("--report-output", required=True)
    dataset_reconcile.add_argument(
        "--labeler-id", default="exact-duplicate-consensus:v1"
    )
    dataset_fuse = dataset_operations.add_parser(
        "fuse",
        help="Fuse source corpora, collapse raw duplicates, and isolate near duplicates.",
    )
    dataset_fuse.set_defaults(handler="data_dataset_fuse")
    dataset_fuse.add_argument("--plan", required=True)
    dataset_harmonize = dataset_operations.add_parser(
        "harmonize",
        help="Align source labels and codes against a pinned terminology repository.",
    )
    dataset_harmonize.set_defaults(handler="data_dataset_harmonize")
    dataset_harmonize.add_argument("--documents", required=True)
    dataset_harmonize.add_argument("--annotations", required=True)
    dataset_harmonize.add_argument("--index", required=True)
    dataset_harmonize.add_argument("--source", action="append", required=True)
    dataset_harmonize.add_argument(
        "--alias-overlay-source", action="append", default=[]
    )
    dataset_harmonize.add_argument("--policy", required=True)
    dataset_harmonize.add_argument("--output", required=True)
    dataset_harmonize.add_argument("--decisions-output", required=True)
    dataset_harmonize.add_argument("--report-output", required=True)
    dataset_curate = dataset_operations.add_parser(
        "curate-annotations",
        help="Partition source annotations into model-eligible and review records.",
    )
    dataset_curate.set_defaults(handler="data_dataset_curate_annotations")
    dataset_curate.add_argument("--annotations", required=True)
    dataset_curate.add_argument("--policy", required=True)
    dataset_curate.add_argument("--accepted-output", required=True)
    dataset_curate.add_argument("--rejected-output", required=True)
    dataset_curate.add_argument("--report-output", required=True)
    dataset_export_spans = dataset_operations.add_parser(
        "export-spans",
        help="Compile source-held-out character spans for local NER training.",
    )
    dataset_export_spans.set_defaults(handler="data_dataset_export_spans")
    dataset_export_spans.add_argument("--documents", required=True)
    dataset_export_spans.add_argument("--annotations", required=True)
    dataset_export_spans.add_argument("--split-manifest", required=True)
    dataset_export_spans.add_argument("--output", required=True)
    dataset_export_spans.add_argument("--manifest-output", required=True)
    dataset_export_spans.add_argument("--entity-type", action="append", default=[])
    dataset_export_spans.add_argument("--max-characters", type=int, default=1600)
    dataset_export_spans.add_argument("--empty-chunk-rate", type=float, default=1.0)
    dataset_export_spans.add_argument("--drop-empty-chunks", action="store_true")

    lexicon = operations.add_parser(
        "lexicon", help="Build mined mention inventories for terminology review."
    )
    lexicon_operations = lexicon.add_subparsers(
        dest="data_lexicon_command", required=True
    )
    lexicon_build = lexicon_operations.add_parser(
        "build", help="Aggregate source mentions without assigning medical codes."
    )
    lexicon_build.set_defaults(handler="data_lexicon_build")
    lexicon_build.add_argument("--documents", required=True)
    lexicon_build.add_argument("--annotations", required=True)
    lexicon_build.add_argument("--output", required=True)
    lexicon_build.add_argument("--conflicts-output", required=True)
    lexicon_build.add_argument("--report-output", required=True)
    lexicon_build.add_argument("--min-occurrences", type=int, default=1)
    lexicon_build.add_argument("--min-documents", type=int, default=1)
    lexicon_build.add_argument(
        "--split-manifest",
        help="Optional frozen snapshot manifest used to select records.",
    )
    lexicon_build.add_argument(
        "--split",
        help="Named split to select; requires --split-manifest.",
    )
    lexicon_crosswalk = lexicon_operations.add_parser(
        "crosswalk",
        help="Query mined mentions against a pinned terminology index.",
    )
    lexicon_crosswalk.set_defaults(handler="data_lexicon_crosswalk")
    lexicon_crosswalk.add_argument("--inventory", required=True)
    lexicon_crosswalk.add_argument("--index", required=True)
    lexicon_crosswalk.add_argument(
        "--source",
        action="append",
        required=True,
        help="Canonical terminology JSONL used to validate the index fingerprint.",
    )
    lexicon_crosswalk.add_argument(
        "--alias-overlay-source",
        action="append",
        default=[],
        help="Alias-overlay JSONL used to validate the complete index fingerprint.",
    )
    lexicon_crosswalk.add_argument("--policy", required=True)
    lexicon_crosswalk.add_argument("--output", required=True)
    lexicon_crosswalk.add_argument("--report-output", required=True)
    lexicon_crosswalk.add_argument("--workers", type=int, default=4)
    lexicon_crosswalk.add_argument("--query-limit", type=int, default=1_000)
    lexicon_crosswalk.add_argument("--candidate-output-limit", type=int, default=20)
    lexicon_crosswalk.add_argument(
        "--lexical-fallback",
        action="store_true",
        help=(
            "After an exact miss, emit bounded FTS candidates for human review; "
            "these rows are never promoted automatically."
        ),
    )
    linked_aliases = lexicon_operations.add_parser(
        "propose-linked-aliases",
        help="Aggregate source annotations with explicit concept links into alias proposals.",
    )
    linked_aliases.set_defaults(handler="data_lexicon_propose_linked_aliases")
    linked_aliases.add_argument("--documents", required=True)
    linked_aliases.add_argument("--annotations", required=True)
    linked_aliases.add_argument("--artifacts", required=True)
    linked_aliases.add_argument("--policy", required=True)
    linked_aliases.add_argument("--output", required=True)
    linked_aliases.add_argument("--decisions-output", required=True)
    linked_aliases.add_argument("--report-output", required=True)

    mapping = operations.add_parser(
        "mapping", help="Compile source crosswalk releases into queryable indexes."
    )
    mapping_operations = mapping.add_subparsers(
        dest="data_mapping_command", required=True
    )
    mapping_dailymed = mapping_operations.add_parser(
        "compile-dailymed-rxnorm",
        help="Deduplicate and index the official SPL-to-RxNorm mapping archive.",
    )
    mapping_dailymed.set_defaults(handler="data_mapping_compile_dailymed_rxnorm")
    mapping_dailymed.add_argument("--artifacts", required=True)
    mapping_dailymed.add_argument("--store", required=True)
    mapping_dailymed.add_argument("--output", required=True)
    mapping_dailymed.add_argument("--index-output", required=True)
    mapping_dailymed.add_argument("--report-output", required=True)

    mapping_audit = mapping_operations.add_parser(
        "audit-dailymed-rxnorm",
        help="Compare a compiled DailyMed mapping with a pinned terminology index.",
    )
    mapping_audit.set_defaults(handler="data_mapping_audit_dailymed_rxnorm")
    mapping_audit.add_argument("--index", required=True)
    mapping_audit.add_argument("--terminology-index", required=True)
    mapping_audit.add_argument(
        "--source",
        action="append",
        required=True,
        help="Canonical terminology JSONL used to validate the index fingerprint.",
    )
    mapping_audit.add_argument("--proposals-output", required=True)
    mapping_audit.add_argument("--report-output", required=True)

    ontology = operations.add_parser(
        "ontology",
        help="Compile pinned ontology releases into terminology and graph artifacts.",
    )
    ontology_operations = ontology.add_subparsers(
        dest="data_ontology_command",
        required=True,
    )
    ontology_obo = ontology_operations.add_parser(
        "compile-obo",
        help="Stream one OBO Graph JSON namespace into rich concepts and canonical IS_A edges.",
    )
    ontology_obo.set_defaults(handler="data_ontology_compile_obo")
    ontology_obo.add_argument("--input", required=True)
    ontology_obo.add_argument("--output-dir", required=True)
    ontology_obo.add_argument("--source-id", required=True)
    ontology_obo.add_argument("--source-version", required=True)
    ontology_obo.add_argument("--iri-prefix", required=True)
    ontology_obo.add_argument(
        "--code-system",
        required=True,
        choices=tuple(system.value for system in CodeSystem),
    )
    ontology_obo.add_argument(
        "--entity-type",
        required=True,
        choices=tuple(entity_type.value for entity_type in EntityType),
    )

    knowledge = operations.add_parser(
        "knowledge",
        help="Promote mined evidence into strict runtime knowledge artifacts.",
    )
    knowledge_operations = knowledge.add_subparsers(
        dest="data_knowledge_command",
        required=True,
    )
    knowledge_aliases = knowledge_operations.add_parser(
        "compile-aliases",
        help="Compile conflict-free aliases against a pinned terminology index.",
    )
    knowledge_aliases.set_defaults(handler="data_knowledge_compile_aliases")
    knowledge_aliases.add_argument("--proposals", action="append", required=True)
    knowledge_aliases.add_argument("--index", required=True)
    knowledge_aliases.add_argument("--source", action="append", required=True)
    knowledge_aliases.add_argument("--base-alias-overlay", action="append", default=[])
    knowledge_aliases.add_argument("--policy", required=True)
    knowledge_aliases.add_argument("--overlay-output", required=True)
    knowledge_aliases.add_argument("--recognition-output", required=True)
    knowledge_aliases.add_argument("--decisions-output", required=True)
    knowledge_aliases.add_argument("--report-output", required=True)
    knowledge_recognition = knowledge_operations.add_parser(
        "compile-recognition",
        help="Compile split-frozen mention inventories into code-free NER concepts.",
    )
    knowledge_recognition.set_defaults(
        handler="data_knowledge_compile_recognition"
    )
    knowledge_recognition.add_argument("--inventory", required=True)
    knowledge_recognition.add_argument("--policy", required=True)
    knowledge_recognition.add_argument(
        "--baseline-dictionary",
        action="append",
        default=[],
        help="Existing recognition dictionary used to reject duplicate/type-conflicting aliases.",
    )
    knowledge_recognition.add_argument("--output", required=True)
    knowledge_recognition.add_argument("--decisions-output", required=True)
    knowledge_recognition.add_argument("--report-output", required=True)
    recognition_benchmark = knowledge_operations.add_parser(
        "benchmark-recognition",
        help="Compare a compact mined dictionary with a baseline entity matcher.",
    )
    recognition_benchmark.set_defaults(handler="data_knowledge_benchmark_recognition")
    recognition_benchmark.add_argument("--documents", required=True)
    recognition_benchmark.add_argument("--annotations", required=True)
    recognition_benchmark.add_argument("--baseline-dictionary", required=True)
    recognition_benchmark.add_argument("--additional-dictionary", required=True)
    recognition_benchmark.add_argument("--entity-type", action="append", required=True)
    recognition_benchmark.add_argument("--output", required=True)
    recognition_benchmark.add_argument(
        "--split-manifest",
        help="Optional frozen snapshot manifest used to select evaluation records.",
    )
    recognition_benchmark.add_argument(
        "--split",
        help="Named evaluation split; requires --split-manifest.",
    )
    graph_compile = knowledge_operations.add_parser(
        "compile-graph",
        help="Deduplicate canonical ontology and mined relation evidence.",
    )
    graph_compile.set_defaults(handler="data_knowledge_compile_graph")
    graph_compile.add_argument("--terminology-source", action="append", default=[])
    graph_compile.add_argument("--alias-overlay", action="append", default=[])
    graph_compile.add_argument("--documents")
    graph_compile.add_argument("--annotations")
    graph_compile.add_argument("--relations")
    graph_compile.add_argument("--entity-type", action="append", default=[])
    graph_compile.add_argument(
        "--accepted-layer",
        action="append",
        choices=("bronze", "silver", "gold", "challenge"),
        default=[],
        help=(
            "Explicit annotation/relation layers for an experiment. When omitted, "
            "the fail-closed graph defaults remain active."
        ),
    )
    graph_compile.add_argument(
        "--accepted-review-status",
        action="append",
        choices=("proposed", "accepted", "rejected"),
        default=[],
        help="Explicit review statuses; defaults to proposed and accepted.",
    )
    graph_compile.add_argument("--linked-only", action="store_true")
    graph_compile.add_argument(
        "--relation-endpoints-only",
        action="store_true",
        help="Exclude unrelated annotations while retaining relation endpoint evidence.",
    )
    graph_compile.add_argument(
        "--canonical-concepts-only",
        action="store_true",
        help="Reject linked annotations whose code is absent from canonical terminology.",
    )
    graph_compile.add_argument(
        "--no-structured-terminology-relations",
        action="store_true",
        help="Keep canonical concepts but skip RxNorm attribute edges.",
    )
    graph_compile.add_argument("--nodes-output", required=True)
    graph_compile.add_argument("--edges-output", required=True)
    graph_compile.add_argument("--evidence-output", required=True)
    graph_compile.add_argument("--report-output", required=True)

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
    label_propose.add_argument(
        "--hosted",
        action="store_true",
        help="Mark the plugin as hosted and enforce document privacy policy.",
    )
    label_propose.add_argument("--output", required=True)
    label_propose.add_argument("--batch-size", type=int, default=16)

    relation = operations.add_parser(
        "relation", help="Run a local relation-labeler adapter."
    )
    relation_operations = relation.add_subparsers(
        dest="data_relation_command", required=True
    )
    relation_propose = relation_operations.add_parser(
        "propose", help="Generate provenance-bearing relation proposals."
    )
    relation_propose.set_defaults(handler="data_relation_propose")
    relation_propose.add_argument("--documents", required=True)
    relation_propose.add_argument("--annotations", required=True)
    relation_propose.add_argument(
        "--adapter", required=True, help="Local factory in module:attribute form."
    )
    relation_propose.add_argument(
        "--adapter-config", help="YAML/JSON factory config mapping."
    )
    relation_propose.add_argument("--output", required=True)
    relation_cooccurrence = relation_operations.add_parser(
        "mine-cooccurrence",
        help=(
            "Mine non-causal same-sentence evidence from a source-pinned training slice."
        ),
    )
    relation_cooccurrence.set_defaults(handler="data_relation_mine_cooccurrence")
    relation_cooccurrence.add_argument("--documents", required=True)
    relation_cooccurrence.add_argument("--annotations", required=True)
    relation_cooccurrence.add_argument("--policy", required=True)
    relation_cooccurrence.add_argument("--output", required=True)
    relation_cooccurrence.add_argument("--report-output", required=True)
    relation_cooccurrence.add_argument(
        "--split-manifest",
        help="Optional frozen snapshot manifest applied in addition to source policy.",
    )
    relation_cooccurrence.add_argument(
        "--split",
        help="Named frozen split; requires --split-manifest.",
    )

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
    review_quality = review_operations.add_parser(
        "quality", help="Measure pairwise gold-review agreement."
    )
    review_quality.set_defaults(handler="data_review_quality")
    review_quality.add_argument("--documents", required=True)
    review_quality.add_argument("--proposals", required=True)
    review_quality.add_argument("--relations")
    review_quality.add_argument("--output", required=True)

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
    snapshot_freeze.add_argument(
        "--source-fingerprint",
        action="append",
        default=[],
        help="Pinned lowercase SHA-256 for a non-artifact upstream manifest.",
    )
    snapshot_freeze.add_argument("--version", required=True)
    snapshot_freeze.add_argument("--created-at", required=True)
    snapshot_freeze.add_argument("--output-dir", required=True)
    snapshot_freeze.add_argument("--development-fraction", type=float, default=0.1)
    snapshot_freeze.add_argument("--development-source", action="append", default=[])
    snapshot_freeze.add_argument("--challenge-source", action="append", default=[])
    snapshot_freeze.add_argument("--challenge-template", action="append", default=[])
    snapshot_freeze.add_argument("--hash-salt", default="medical-kg-snapshot-v1")
    snapshot_freeze.add_argument("--max-synthetic-fraction", type=float, default=0.4)
    snapshot_freeze.add_argument("--manifest-only", action="store_true")
    snapshot_freeze.add_argument("--skip-agreement-gate", action="store_true")

    run = operations.add_parser("run", help="Run a resumable declarative mining plan.")
    run.set_defaults(handler="data_run")
    run.add_argument("--plan", required=True)
