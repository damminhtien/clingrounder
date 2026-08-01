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
        help=("Pipeline prediction JSONL required by predicted_ner_exact_unique mode."),
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
    run.add_argument(
        "--parallel-backend", choices=("serial", "thread", "process"), default="process"
    )
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
    phase1 = plugins.add_parser("phase1", help="Run Phase 1 benchmark workflows.")
    operations = phase1.add_subparsers(dest="phase1_command", required=True)

    submission = operations.add_parser(
        "submission",
        help="Build and release-validate a strict submission ZIP.",
    )
    submission.set_defaults(handler="benchmark_phase1_submission")
    source = submission.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir")
    source.add_argument(
        "--documents",
        help="Private mined documents.jsonl with source-document metadata.",
    )
    submission.add_argument(
        "--source-archive-sha256",
        help="Required pinned archive SHA-256 when --documents is used.",
    )
    submission.add_argument("--output-dir", default="output")
    submission.add_argument("--zip", default="output.zip")
    submission.add_argument(
        "--run-root",
        help="Create a unique content-hashed run below this directory.",
    )
    submission.add_argument("--run-label", default="phase1")
    submission.add_argument(
        "--provenance-input",
        action="append",
        default=[],
        help="Additional immutable input, such as a trained-model manifest; repeat as needed.",
    )
    submission.add_argument("--dictionary", default="data/dictionaries/seed_concepts.jsonl")
    submission.add_argument("--abbreviations", default="data/dictionaries/abbreviations.jsonl")
    submission.add_argument(
        "--pipeline-config",
        help=(
            "Optional reusable pipeline profile. Its mined recognition sources and full "
            "terminology index are preserved while benchmark paths override the base files."
        ),
    )
    submission.add_argument(
        "--validation-dictionary",
        dest="validation_dictionaries",
        action="append",
        default=[],
        help=(
            "Canonical JSONL source accepted by release validation; repeat for TT06 and "
            "RxNorm when a full terminology profile emits both systems."
        ),
    )
    submission.add_argument(
        "--assertion-policy",
        choices=("empty", "pipeline"),
        default="pipeline",
    )
    submission.add_argument(
        "--candidate-policy",
        choices=("empty", "pipeline"),
        default="pipeline",
    )
    submission.add_argument("--max-candidates", type=int, default=5)
    submission.add_argument(
        "--parallel-backend", choices=("serial", "thread", "process"), default="process"
    )
    submission.add_argument("--workers", type=int, default=1)
    submission.add_argument("--chunksize", type=int, default=4)

    round2 = operations.add_parser(
        "round2",
        help="Inspect private Round 2 input without creating annotation memory.",
    )
    round2_operations = round2.add_subparsers(dest="phase1_round2_command", required=True)
    audit = round2_operations.add_parser(
        "audit",
        help="Write aggregate profile, duplicate evidence, and novelty queue.",
    )
    audit.set_defaults(handler="benchmark_phase1_round2_audit")
    audit.add_argument("--documents", required=True)
    audit.add_argument("--reference-input-dir", default="data/raw/input")
    audit.add_argument("--reference-gold-dir", default="data/manual_gold")
    audit.add_argument(
        "--reference-split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )
    audit.add_argument("--output-dir", required=True)

    probes = round2_operations.add_parser(
        "probes",
        help="Build isolated assertion and region-routed entity probe ZIPs.",
    )
    probes.set_defaults(handler="benchmark_phase1_round2_probes")
    probes.add_argument("--documents", required=True)
    probes.add_argument("--source-archive-sha256", required=True)
    probes.add_argument("--base", required=True)
    probes.add_argument("--expected-base-sha256", required=True)
    probes.add_argument(
        "--source",
        action="append",
        default=[],
        help="Calibrated proposal artifact as NAME=DIR_OR_ZIP; repeat for Qwen/XLM-R.",
    )
    probes.add_argument(
        "--build-full-source",
        action="append",
        default=[],
        help=(
            "Canonicalize the complete named --source projection and build its "
            "A_NEG_HIST combination; repeat by source name."
        ),
    )
    probes.add_argument(
        "--build-consensus-source",
        action="append",
        default=[],
        help=(
            "Build an all-region additive variant for a named --source whose producer "
            "already enforced independent evidence agreement; repeat by source name."
        ),
    )
    probes.add_argument(
        "--dictionary",
        default="data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl",
    )
    probes.add_argument(
        "--validation-dictionary",
        dest="validation_dictionaries",
        action="append",
        default=[
            "data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl",
            "data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl",
        ],
        help="Canonical code source used by strict probe validation; repeat as needed.",
    )
    probes.add_argument("--output-root", default="outputs/phase1/round2")
    probes.add_argument("--run-label", default="round2-breakthrough-probes")
    probes.add_argument("--expected-count", type=int, default=100)
    probes.add_argument("--minimum-agreement-sources", type=int, default=2)
    probes.add_argument(
        "--no-expand-repeated-mentions",
        action="store_true",
        help="Do not recover repeated exact mentions in proposal sources.",
    )
    probes.add_argument(
        "--candidate-probe",
        action="append",
        choices=(
            "icd_top1_keep_rx",
            "rx_only",
            "rx_unique_only",
            "rx_unique_keep_icd",
        ),
        default=[],
        help=(
            "Build a candidate-only abstention probe on the frozen baseline; repeat for "
            "icd_top1_keep_rx, rx_only, rx_unique_only, and rx_unique_keep_icd."
        ),
    )
    probes.add_argument(
        "--reviewed-rxnorm-map",
        help=(
            "Reviewed candidate JSONL used to fill empty medication candidates by "
            "exact normalized mention."
        ),
    )
    probes.add_argument(
        "--reviewed-rxnorm-min-occurrence-support",
        type=int,
        default=2,
        help="Minimum reviewed occurrence support for an exact RxNorm mapping.",
    )
    probes.add_argument(
        "--reviewed-rxnorm-min-document-support",
        type=int,
        default=1,
        help="Minimum reviewed document support for an exact RxNorm mapping.",
    )
    probes.add_argument(
        "--structured-rxnorm-fill-empty",
        action="store_true",
        help=(
            "Fill only empty medication candidates when exact/toneless structured RxNorm "
            "retrieval yields one high-confidence code."
        ),
    )
    probes.add_argument(
        "--structured-rxnorm-minimum-score",
        type=float,
        default=0.95,
        help="Minimum reranker score for --structured-rxnorm-fill-empty.",
    )

    proposal_verify = round2_operations.add_parser(
        "proposal-verifier",
        help="Apply a frozen calibrated verifier as an additive-only entity probe.",
    )
    proposal_verify.set_defaults(
        handler="benchmark_phase1_round2_proposal_verifier"
    )
    proposal_verify.add_argument("--documents", required=True)
    proposal_verify.add_argument("--source-archive-sha256", required=True)
    proposal_verify.add_argument("--base", required=True)
    proposal_verify.add_argument("--expected-base-sha256", required=True)
    proposal_verify.add_argument("--proposal-source", required=True)
    proposal_verify.add_argument("--expected-proposal-source-sha256", required=True)
    proposal_verify.add_argument("--verifier", required=True)
    proposal_verify.add_argument("--expected-verifier-sha256", required=True)
    proposal_verify.add_argument(
        "--dictionary",
        default="data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl",
    )
    proposal_verify.add_argument(
        "--validation-dictionary",
        dest="validation_dictionaries",
        action="append",
        default=[
            "data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl",
            "data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl",
        ],
    )
    proposal_verify.add_argument("--output-root", default="outputs/phase1/round2")
    proposal_verify.add_argument("--run-label", default="round2-proposal-verifier")
    proposal_verify.add_argument("--expected-count", type=int, default=100)

    max_score = round2_operations.add_parser(
        "max-score",
        help="Compose pinned Rule/XLM-R/Qwen/VietMed artifacts under a verified model budget.",
    )
    max_score.set_defaults(handler="benchmark_phase1_round2_max_score")
    max_score.add_argument(
        "--config",
        required=True,
        help="Pinned Phase 1 max-score run specification.",
    )

    proposal_matrix = operations.add_parser(
        "proposal-matrix",
        help="Align rule/model/LLM/support proposals and retain source confidence evidence.",
    )
    proposal_matrix.set_defaults(handler="benchmark_phase1_proposal_matrix")
    proposal_matrix.add_argument(
        "--target-source",
        action="append",
        default=[],
        help="Flat Phase 1 target source as NAME=DIR_OR_ZIP; repeat as needed.",
    )
    proposal_matrix.add_argument(
        "--internal-source",
        action="append",
        default=[],
        help="Internal prediction JSONL as NAME=PATH; repeat as needed.",
    )
    proposal_matrix.add_argument(
        "--compatible-source",
        action="append",
        default=[],
        help="Support-only compatible source as NAME=DIR_OR_ZIP_OR_JSONL.",
    )
    proposal_matrix.add_argument("--input-dir", default="data/raw/input")
    proposal_matrix.add_argument("--output-dir", required=True)

    proposal_calibrate = operations.add_parser(
        "proposal-calibrate",
        help="Train a leakage-safe calibrated verifier over aligned entity proposals.",
    )
    proposal_calibrate.set_defaults(
        handler="benchmark_phase1_proposal_calibrate"
    )
    proposal_calibrate.add_argument("--matrix", required=True)
    proposal_calibrate.add_argument("--input-dir", default="data/raw/input")
    proposal_calibrate.add_argument("--gold-dir", default="data/manual_gold")
    proposal_calibrate.add_argument(
        "--model-split-manifest",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-v1/split_manifest.json"
        ),
    )
    proposal_calibrate.add_argument(
        "--frozen-split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )
    proposal_calibrate.add_argument(
        "--training-governance",
        help=(
            "Strict policy that authorizes all reviewed labels for final fitting. "
            "Omit to preserve the legacy sealed-holdout diagnostic."
        ),
    )
    proposal_calibrate.add_argument(
        "--source-role",
        action="append",
        required=True,
        help="Proposal source and portable role as NAME=rule|llm|token_model|ensemble|verifier.",
    )
    proposal_calibrate.add_argument(
        "--minimum-development-precision",
        type=float,
        help=(
            "Select the highest-recall per-type threshold meeting this precision; "
            "omit to maximize development F1."
        ),
    )
    proposal_calibrate.add_argument(
        "--fit-mode",
        choices=("development", "full_oof"),
        default="development",
        help=(
            "development preserves the legacy split diagnostic; full_oof fits every "
            "supplied labeled proposal and derives thresholds from grouped OOF predictions."
        ),
    )
    proposal_calibrate.add_argument("--output-dir", required=True)

    proposal_resolve = operations.add_parser(
        "proposal-resolve",
        help="Replace proposal union with calibrated probability and overlap resolution.",
    )
    proposal_resolve.set_defaults(handler="benchmark_phase1_proposal_resolve")
    proposal_resolve.add_argument("--matrix", required=True)
    proposal_resolve.add_argument("--verifier", required=True)
    proposal_resolve.add_argument(
        "--source-role",
        action="append",
        required=True,
        help="Proposal source and portable role as NAME=rule|llm|token_model|ensemble|verifier.",
    )
    proposal_resolve.add_argument("--input-dir", default="data/raw/input")
    proposal_resolve.add_argument("--output-dir", required=True)

    boundary_calibrate = operations.add_parser(
        "boundary-calibrate",
        help="Train a proposal-conditioned ranker over raw boundary alternatives.",
    )
    boundary_calibrate.set_defaults(
        handler="benchmark_phase1_boundary_calibrate"
    )
    boundary_input = boundary_calibrate.add_mutually_exclusive_group(required=True)
    boundary_input.add_argument(
        "--matrix",
        help="Proposal matrix from which boundary candidates will be generated.",
    )
    boundary_input.add_argument(
        "--dataset-dir",
        help="Previously materialized boundary dataset to train without regeneration.",
    )
    boundary_calibrate.add_argument(
        "--proposal-verifier",
        help="Optional frozen proposal verifier used as one boundary feature.",
    )
    boundary_calibrate.add_argument(
        "--fit-mode",
        choices=["development", "full_oof"],
        default="development",
        help="Use full_oof only with governed all-manual-gold supervision.",
    )
    boundary_calibrate.add_argument(
        "--training-governance",
        help="Required by full_oof before opening the legacy holdout labels.",
    )
    boundary_calibrate.add_argument(
        "--dictionary",
        action="append",
        default=[],
        help="Recognition JSONL used to generate trie boundary alternatives; repeatable.",
    )
    boundary_calibrate.add_argument("--input-dir", default="data/raw/input")
    boundary_calibrate.add_argument("--gold-dir", default="data/manual_gold")
    boundary_calibrate.add_argument(
        "--model-split-manifest",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-v1/split_manifest.json"
        ),
    )
    boundary_calibrate.add_argument(
        "--frozen-split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )
    boundary_calibrate.add_argument(
        "--source-role",
        action="append",
        default=[],
        help="Proposal source and portable role as NAME=ROLE.",
    )
    boundary_calibrate.add_argument("--output-dir", required=True)

    boundary_resolve = operations.add_parser(
        "boundary-resolve",
        help="Rank generated boundary families and write raw-offset Phase 1 rows.",
    )
    boundary_resolve.set_defaults(handler="benchmark_phase1_boundary_resolve")
    boundary_resolve.add_argument("--matrix", required=True)
    boundary_resolve.add_argument("--boundary-verifier", required=True)
    boundary_resolve.add_argument(
        "--proposal-verifier",
        help="Required when the boundary verifier was trained with base probabilities.",
    )
    boundary_resolve.add_argument(
        "--dictionary",
        action="append",
        default=[],
        help="Same recognition JSONL set used during boundary training; repeatable.",
    )
    boundary_resolve.add_argument(
        "--source-role",
        action="append",
        required=True,
        help="Proposal source and portable role as NAME=ROLE.",
    )
    boundary_resolve.add_argument("--input-dir", default="data/raw/input")
    boundary_resolve.add_argument("--output-dir", required=True)

    proposal_score = operations.add_parser(
        "proposal-score",
        help="Score frozen rule/model/LLM/support proposal sources on train and development.",
    )
    proposal_score.set_defaults(handler="benchmark_phase1_proposal_score")
    proposal_score.add_argument(
        "--target-source",
        action="append",
        default=[],
        help="Flat Phase 1 target source as NAME=DIR_OR_ZIP; repeat as needed.",
    )
    proposal_score.add_argument(
        "--internal-source",
        action="append",
        default=[],
        help="Internal prediction JSONL as NAME=PATH; repeat as needed.",
    )
    proposal_score.add_argument(
        "--compatible-source",
        action="append",
        default=[],
        help="Support-only compatible source as NAME=DIR_OR_ZIP_OR_JSONL.",
    )
    proposal_score.add_argument("--input-dir", default="data/raw/input")
    proposal_score.add_argument("--gold-dir", default="data/manual_gold")
    proposal_score.add_argument(
        "--model-split-manifest",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-v1/split_manifest.json"
        ),
    )
    proposal_score.add_argument(
        "--frozen-split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )
    proposal_score.add_argument("--output-dir", required=True)

    type_verifier = operations.add_parser(
        "type-verifier",
        help="Train the abstaining DISEASE/SYMPTOM/NONE verifier.",
    )
    type_verifier.set_defaults(handler="benchmark_phase1_type_verifier")
    type_verifier.add_argument("--matrix", required=True)
    type_verifier.add_argument(
        "--representation-source",
        help=(
            "Optional support-only directory/ZIP whose source labels become representation "
            "features, never target labels."
        ),
    )
    type_verifier.add_argument("--input-dir", default="data/raw/input")
    type_verifier.add_argument("--gold-dir", default="data/manual_gold")
    type_verifier.add_argument(
        "--model-split-manifest",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-v1/split_manifest.json"
        ),
    )
    type_verifier.add_argument(
        "--frozen-split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )
    type_verifier.add_argument("--output-dir", required=True)

    golden = round2_operations.add_parser(
        "golden",
        help="Build inferred strict/review labels from independent proposal sources.",
    )
    golden.set_defaults(handler="benchmark_phase1_round2_golden")
    golden.add_argument("--documents", required=True)
    golden.add_argument("--source-archive-sha256", required=True)
    golden.add_argument(
        "--source",
        action="append",
        required=True,
        help="Independent proposal artifact as NAME=DIR_OR_ZIP; repeat at least twice.",
    )
    golden.add_argument(
        "--dictionary",
        default="data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl",
    )
    golden.add_argument(
        "--validation-dictionary",
        dest="validation_dictionaries",
        action="append",
        default=[
            "data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl",
            "data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl",
        ],
        help="Pinned terminology source for candidate filtering; repeat as needed.",
    )
    golden.add_argument("--output-root", default="outputs/phase1/round2_golden")
    golden.add_argument("--run-label", default="round2-golden-local")
    golden.add_argument("--expected-count", type=int, default=100)
    golden.add_argument("--minimum-sources", type=int, default=2)
    golden.add_argument(
        "--btc-example-input",
        default="tests/fixtures/phase1/btc_medication_list_crlf.txt",
    )
    golden.add_argument(
        "--btc-example-output",
        default="tests/fixtures/phase1/btc_medication_list_expected.json",
    )

    model_data = operations.add_parser(
        "model-data",
        help="Build leakage-safe Phase 1 model supervision.",
    )
    model_data_operations = model_data.add_subparsers(
        dest="phase1_model_data_command",
        required=True,
    )
    model_data_build = model_data_operations.add_parser(
        "build",
        help="Build the five-type NER train/development dataset.",
    )
    model_data_build.set_defaults(handler="benchmark_phase1_model_data_build")
    model_data_build.add_argument("--input-dir", default="data/raw/input")
    model_data_build.add_argument("--gold-dir", default="data/manual_gold")
    model_data_build.add_argument(
        "--frozen-split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )
    model_data_build.add_argument(
        "--public-spec-input",
        default="tests/fixtures/phase1/btc_medication_list_crlf.txt",
    )
    model_data_build.add_argument(
        "--public-spec-expected",
        default="tests/fixtures/phase1/btc_medication_list_expected.json",
    )
    model_data_build.add_argument("--output-dir", required=True)
    model_data_build.add_argument("--development-fraction", type=float, default=0.2)
    model_data_build.add_argument("--split-salt", default="42")
    model_data_build.add_argument("--max-characters", type=int, default=1600)
    model_data_build.add_argument("--empty-chunk-rate", type=float, default=1.0)
    model_data_build.add_argument(
        "--exclude-empty-chunks",
        action="store_true",
        help="Drop chunks without an entity after deterministic chunking.",
    )

    model_data_final_fit = model_data_operations.add_parser(
        "build-final-fit",
        help="Build five-type NER supervision from all authorized final-fit Phase 1 records.",
    )
    model_data_final_fit.set_defaults(handler="benchmark_phase1_model_data_build_final_fit")
    model_data_final_fit.add_argument("--output-dir", required=True)
    model_data_final_fit.add_argument(
        "--training-governance",
        default="configs/models/phase1-training-governance-2026-07-30.yaml",
    )
    model_data_final_fit.add_argument(
        "--model-split-manifest",
        default="outputs/mining/model-datasets/phase1-manual-five-type-v1/split_manifest.json",
    )
    model_data_final_fit.add_argument(
        "--frozen-split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )
    model_data_final_fit.add_argument("--manual-input-dir", default="data/raw/input")
    model_data_final_fit.add_argument("--manual-gold-dir", default="data/manual_gold")
    model_data_final_fit.add_argument(
        "--authorized-archive",
        help="Owner-authorized ground-truth archive; otherwise use the governed environment variable.",
    )
    model_data_final_fit.add_argument("--max-characters", type=int, default=1600)
    model_data_final_fit.add_argument("--empty-chunk-rate", type=float, default=1.0)
    model_data_final_fit.add_argument(
        "--exclude-empty-chunks",
        action="store_true",
        help="Drop chunks without an entity after deterministic chunking.",
    )

    model_data_augment = model_data_operations.add_parser(
        "augment-regions",
        help="Add bounded train-only Q&A and educational discourse views.",
    )
    model_data_augment.set_defaults(
        handler="benchmark_phase1_model_data_augment_regions"
    )
    model_data_augment.add_argument(
        "--source-dataset",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-v1/spans.jsonl"
        ),
    )
    model_data_augment.add_argument(
        "--source-manifest",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-v1/manifest.json"
        ),
    )
    model_data_augment.add_argument(
        "--source-build-manifest",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-v1/build_manifest.json"
        ),
    )
    model_data_augment.add_argument("--output-dir", required=True)
    model_data_augment.add_argument(
        "--max-synthetic-fraction",
        type=float,
        default=0.4,
        help="Maximum synthetic share after augmentation; hard-capped at 0.4.",
    )
    model_data_augment.add_argument(
        "--seed",
        default="phase1-qa-educational-v1",
    )

    model_data_synthetic = model_data_operations.add_parser(
        "augment-user-synthetic",
        help="Validate and add a bounded train-only user synthetic archive.",
    )
    model_data_synthetic.set_defaults(
        handler="benchmark_phase1_model_data_augment_user_synthetic"
    )
    model_data_synthetic.add_argument("--archive", required=True)
    model_data_synthetic.add_argument("--archive-sha256", required=True)
    model_data_synthetic.add_argument(
        "--source-dataset",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-v1/spans.jsonl"
        ),
    )
    model_data_synthetic.add_argument(
        "--source-manifest",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-v1/manifest.json"
        ),
    )
    model_data_synthetic.add_argument("--output-dir", required=True)
    model_data_synthetic.add_argument(
        "--max-synthetic-fraction",
        type=float,
        default=0.4,
        help="Maximum synthetic share of train records; hard-capped at 0.4.",
    )
    model_data_synthetic.add_argument(
        "--seed",
        default="phase1-user-synthetic-balanced-v1",
    )

    qwen_data_build = model_data_operations.add_parser(
        "build-qwen",
        help="Build leakage-safe Qwen extraction and adjudication instructions.",
    )
    qwen_data_build.set_defaults(handler="benchmark_phase1_qwen_data_build")
    qwen_data_build.add_argument(
        "--source-dataset",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-qa-edu-v1/spans.jsonl"
        ),
    )
    qwen_data_build.add_argument(
        "--source-manifest",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-qa-edu-v1/manifest.json"
        ),
    )
    qwen_data_build.add_argument(
        "--hard-negative-predictions",
        help="Optional XLM-R predictions over train documents only.",
    )
    qwen_data_build.add_argument(
        "--output-dir",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-qwen-instructions-v1"
        ),
    )
    qwen_data_build.add_argument(
        "--exclude-development",
        action="store_true",
        help="Write train records only; development remains excluded rather than merged.",
    )
    qwen_data_build.add_argument(
        "--review-masks-per-train-record",
        type=int,
        default=0,
        help="Create zero to four deterministic missing-only reviewer masks per train record.",
    )
    qwen_data_build.add_argument(
        "--review-keep-fraction",
        type=float,
        default=0.5,
        help="Expected share of unique entities shown as already extracted.",
    )
    qwen_data_build.add_argument(
        "--review-seed",
        default="phase1-qwen-review-missing-v1",
    )

    model_data_calibrate = model_data_operations.add_parser(
        "calibrate",
        help="Verify a full model and select per-type thresholds on development.",
    )
    model_data_calibrate.set_defaults(handler="benchmark_phase1_model_data_calibrate")
    model_data_calibrate.add_argument(
        "--pipeline-config",
        default="configs/pipeline/phase1-five-type-model-only.yaml",
        help="Verified model-only pipeline profile.",
    )
    model_data_calibrate.add_argument("--output-dir", required=True)

    model_data_compare = model_data_operations.add_parser(
        "compare",
        help="Compare rule, model, and hybrid NER outputs on development.",
    )
    model_data_compare.set_defaults(handler="benchmark_phase1_model_data_compare")
    _phase1_model_selection_arguments(model_data_compare)
    model_data_compare.add_argument(
        "--variant",
        action="append",
        required=True,
        help="One NAME=DIR_OR_ZIP input; provide rule, model, and hybrid.",
    )
    model_data_compare.add_argument("--output", required=True)
    model_data_compare.add_argument(
        "--open-frozen-holdout",
        action="store_true",
        help="Explicitly score the final frozen holdout after development selection.",
    )

    qwen = operations.add_parser(
        "qwen",
        help="Inspect or execute pinned Qwen proposal runs.",
    )
    qwen_operations = qwen.add_subparsers(
        dest="phase1_qwen_command",
        required=True,
    )
    qwen_inspect = qwen_operations.add_parser(
        "inspect",
        help="Validate model identity, parameter budget, cost cap, and data paths.",
    )
    qwen_inspect.set_defaults(handler="benchmark_phase1_qwen_inspect")
    qwen_inspect.add_argument("--config", required=True)
    qwen_propose = qwen_operations.add_parser(
        "propose",
        help="Run extraction/adjudication or missing-only review on private documents.",
    )
    qwen_propose.set_defaults(handler="benchmark_phase1_qwen_propose")
    qwen_propose.add_argument("--config", required=True)
    qwen_propose.add_argument("--documents", required=True)
    qwen_propose.add_argument("--source-archive-sha256", required=True)
    qwen_propose.add_argument("--output-dir", required=True)
    qwen_propose.add_argument(
        "--support-source",
        action="append",
        default=[],
        help="Optional NAME=DIR_OR_ZIP rule/XLM-R support; XLM-R alone is never emitted.",
    )
    qwen_propose.add_argument(
        "--review-source",
        help=(
            "Optional NAME=DIR_OR_ZIP frozen projection shown to the missing-only reviewer."
        ),
    )
    qwen_propose.add_argument(
        "--review-max-rounds",
        type=int,
        default=2,
        help="Stop after this many missing-only review rounds, or earlier when no entity is added.",
    )
    qwen_propose.add_argument(
        "--review-only",
        action="store_true",
        help="Skip recall, targeted, and adjudication passes to review one frozen source cheaply.",
    )
    qwen_propose.add_argument("--expected-count", type=int, default=100)
    qwen_propose.add_argument("--no-adjudication", action="store_true")
    qwen_propose.add_argument(
        "--extraction-mode",
        choices=("recall_only", "recall_and_targeted"),
        default="recall_and_targeted",
        help=(
            "Run one all-type recall pass, or recall plus five type-targeted passes. "
            "The selected mode is part of the resume fingerprint."
        ),
    )
    qwen_propose.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse complete per-document outputs only when the run fingerprint and "
            "raw offsets still match."
        ),
    )
    qwen_final_supervision = qwen_operations.add_parser(
        "propose-final-supervision",
        help="Run Qwen exact-quote proposal passes over governed final-fit supervision only.",
    )
    qwen_final_supervision.set_defaults(
        handler="benchmark_phase1_qwen_final_supervision_propose"
    )
    qwen_final_supervision.add_argument("--config", required=True)
    qwen_final_supervision.add_argument("--output-dir", required=True)
    qwen_final_supervision.add_argument(
        "--training-governance",
        default="configs/models/phase1-training-governance-2026-07-30.yaml",
    )
    qwen_final_supervision.add_argument(
        "--model-split-manifest",
        default="outputs/mining/model-datasets/phase1-manual-five-type-v1/split_manifest.json",
    )
    qwen_final_supervision.add_argument(
        "--frozen-split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )
    qwen_final_supervision.add_argument("--manual-input-dir", default="data/raw/input")
    qwen_final_supervision.add_argument("--manual-gold-dir", default="data/manual_gold")
    qwen_final_supervision.add_argument(
        "--authorized-archive",
        help="Owner-authorized nested archive; otherwise use the governed environment variable.",
    )
    qwen_final_supervision.add_argument(
        "--extraction-mode",
        choices=("recall_only", "recall_and_targeted"),
        default="recall_and_targeted",
    )
    qwen_final_supervision.add_argument("--resume", action="store_true")

    joint_span = operations.add_parser(
        "joint-span",
        help="Prepare and train learned joint span/type verifier artifacts.",
    )
    joint_span_operations = joint_span.add_subparsers(
        dest="phase1_joint_span_command",
        required=True,
    )
    joint_span_prepare = joint_span_operations.add_parser(
        "prepare-final-fit",
        help=(
            "Build a governed final-fit lattice dataset from RuleNER and independently "
            "materialized model proposal sources."
        ),
    )
    joint_span_prepare.set_defaults(handler="benchmark_phase1_joint_span_prepare_final_fit")
    joint_span_prepare.add_argument(
        "--model-source",
        action="append",
        required=True,
        help="Pinned proposal artifact as NAME=DIRECTORY; repeat for Qwen/XLM-R.",
    )
    joint_span_prepare.add_argument(
        "--source-role",
        action="append",
        required=True,
        help="Role for every model source as NAME=llm|token_model|ensemble.",
    )
    joint_span_prepare.add_argument(
        "--dictionary",
        action="append",
        default=[
            "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl",
            "data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl",
            "data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl",
        ],
        help="Canonical recognition dictionaries; repeat to add a pinned source.",
    )
    joint_span_prepare.add_argument("--output-dir", required=True)
    joint_span_prepare.add_argument(
        "--training-governance",
        default="configs/models/phase1-training-governance-2026-07-30.yaml",
    )
    joint_span_prepare.add_argument(
        "--model-split-manifest",
        default="outputs/mining/model-datasets/phase1-manual-five-type-v1/split_manifest.json",
    )
    joint_span_prepare.add_argument(
        "--frozen-split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )
    joint_span_prepare.add_argument("--manual-input-dir", default="data/raw/input")
    joint_span_prepare.add_argument("--manual-gold-dir", default="data/manual_gold")
    joint_span_prepare.add_argument(
        "--authorized-archive",
        help="Owner-authorized ground-truth archive; otherwise use the governed environment variable.",
    )
    joint_span_token_source = joint_span_operations.add_parser(
        "materialize-token-source",
        help="Project one pinned local five-type token checkpoint into a governed source artifact.",
    )
    joint_span_token_source.set_defaults(
        handler="benchmark_phase1_joint_span_materialize_token_source"
    )
    joint_span_token_source.add_argument("--model-path", required=True)
    joint_span_token_source.add_argument("--model-fingerprint", required=True)
    joint_span_token_source.add_argument("--model-id", required=True)
    joint_span_token_source.add_argument("--base-revision", required=True)
    joint_span_token_source.add_argument("--output-dir", required=True)
    joint_span_token_source.add_argument("--source-name", default="xlmr")
    joint_span_token_source.add_argument("--device", default="cpu")
    joint_span_token_source.add_argument("--batch-size", type=int, default=16)
    joint_span_token_source.add_argument("--max-length", type=int, default=512)
    joint_span_token_source.add_argument("--stride", type=int, default=64)
    joint_span_token_source.add_argument(
        "--default-confidence-threshold",
        type=float,
        default=0.0,
    )
    joint_span_token_source.add_argument(
        "--training-governance",
        default="configs/models/phase1-training-governance-2026-07-30.yaml",
    )
    joint_span_token_source.add_argument(
        "--model-split-manifest",
        default="outputs/mining/model-datasets/phase1-manual-five-type-v1/split_manifest.json",
    )
    joint_span_token_source.add_argument(
        "--frozen-split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )
    joint_span_token_source.add_argument("--manual-input-dir", default="data/raw/input")
    joint_span_token_source.add_argument("--manual-gold-dir", default="data/manual_gold")
    joint_span_token_source.add_argument(
        "--authorized-archive",
        help="Owner-authorized ground-truth archive; otherwise use the governed environment variable.",
    )
    joint_span_run = joint_span_operations.add_parser(
        "run",
        help="Resolve pinned raw sources with a learned joint span/type verifier and export a ZIP.",
    )
    joint_span_run.set_defaults(handler="benchmark_phase1_joint_span_run")
    joint_span_run.add_argument("--config", required=True)
    joint_span_train = joint_span_operations.add_parser(
        "train",
        help="Train a pinned transformer verifier from a prepared final-fit lattice dataset.",
    )
    joint_span_train.set_defaults(handler="benchmark_phase1_joint_span_train")
    joint_span_train.add_argument("--dataset", required=True)
    joint_span_train.add_argument("--dataset-manifest", required=True)
    joint_span_train.add_argument("--output-dir", required=True)
    joint_span_train.add_argument("--model-id", required=True)
    joint_span_train.add_argument("--revision", required=True)
    joint_span_train.add_argument("--initialization-model")
    joint_span_train.add_argument("--initialization-fingerprint")
    joint_span_train.add_argument("--max-length", type=int, default=384)
    joint_span_train.add_argument("--train-batch-size", type=int, default=8)
    joint_span_train.add_argument("--evaluation-batch-size", type=int, default=16)
    joint_span_train.add_argument("--epochs", type=float, default=4.0)
    joint_span_train.add_argument("--learning-rate", type=float, default=2e-5)
    joint_span_train.add_argument("--weight-decay", type=float, default=0.01)
    joint_span_train.add_argument("--warmup-ratio", type=float, default=0.08)
    joint_span_train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    joint_span_train.add_argument("--seed", type=int, default=42)
    joint_span_train.add_argument("--fp16", action="store_true")
    joint_span_train.add_argument("--bf16", action="store_true")
    joint_span_train.add_argument("--use-cpu", action="store_true")
    joint_span_train.add_argument("--cache-dir")
    joint_span_train.add_argument("--overwrite-output", action="store_true")
    qwen_support = qwen_operations.add_parser(
        "build-vietnamese-support",
        help="Run a pinned Vietnamese NER model as support that Qwen must confirm.",
    )
    qwen_support.set_defaults(handler="benchmark_phase1_qwen_vietnamese_support")
    qwen_support.add_argument("--config", required=True)
    qwen_support.add_argument("--documents", required=True)
    qwen_support.add_argument("--source-archive-sha256", required=True)
    qwen_support.add_argument("--output-dir", required=True)
    qwen_support.add_argument("--expected-count", type=int, default=100)


def _model_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    model = commands.add_parser("model", help="Validate datasets and train local models.")
    operations = model.add_subparsers(dest="model_command", required=True)

    inspect_budget = operations.add_parser(
        "inspect-inference-budget",
        help="Verify exact active parameter counts and reserved model capacity.",
    )
    inspect_budget.set_defaults(handler="model_inspect_inference_budget")
    inspect_budget.add_argument("--config", required=True)
    inspect_budget.add_argument("--output")

    build_dapt = operations.add_parser(
        "build-dapt-corpus",
        help="Build source-pinned MLM lanes, including an isolated Round 2 lane.",
    )
    build_dapt.set_defaults(handler="model_build_dapt_corpus")
    build_dapt.add_argument("--config", required=True)

    inspect_dapt = operations.add_parser(
        "inspect-xlmr-dapt-run",
        help="Verify joint MLM/contrastive inputs and Round 2 provenance without ML imports.",
    )
    inspect_dapt.set_defaults(handler="model_inspect_xlmr_dapt_run")
    inspect_dapt.add_argument("--config", required=True)

    train_dapt = operations.add_parser(
        "train-xlmr-dapt-run",
        help="Validate Linux/CUDA and train one pinned joint XLM-R DAPT run.",
    )
    train_dapt.set_defaults(handler="model_train_xlmr_dapt_run")
    train_dapt.add_argument("--config", required=True)
    train_dapt.add_argument("--resume-from-checkpoint")
    train_dapt.add_argument(
        "--max-steps",
        type=int,
        help="Smoke-only optimizer-step override; output is marked non-promotable.",
    )
    train_dapt.add_argument(
        "--output-dir",
        help="Run-root-relative smoke output; the checked-in training path is unchanged.",
    )

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
    train.add_argument(
        "--unaligned-span-policy",
        choices=("error", "mask"),
        default="error",
        help=(
            "Fail on tokenizer-inexpressible gold boundaries, or mask crossing tokens from loss."
        ),
    )
    train.add_argument(
        "--cpu-smoke-text",
        help=(
            "Raw UTF-8 text file used to reload the saved model and validate inference offsets; "
            "requires --cpu and marks the run as non-submittable."
        ),
    )

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

    inspect_qlora = operations.add_parser(
        "inspect-causal-qlora-run",
        help="Validate a pinned QLoRA run spec without ML imports.",
    )
    inspect_qlora.set_defaults(handler="model_inspect_causal_qlora_run")
    inspect_qlora.add_argument("--config", required=True)

    finalize_qlora = operations.add_parser(
        "finalize-causal-qlora-run",
        help="Verify and finalize a completed QLoRA artifact from a source bundle.",
    )
    finalize_qlora.set_defaults(handler="model_finalize_causal_qlora_run")
    finalize_qlora.add_argument("--config", required=True)

    train_qlora = operations.add_parser(
        "train-causal-qlora-run",
        help="Validate Linux/CUDA and execute one pinned QLoRA stage.",
    )
    train_qlora.set_defaults(handler="model_train_causal_qlora_run")
    train_qlora.add_argument("--config", required=True)
    train_qlora.add_argument("--resume-from-checkpoint")
    train_qlora.add_argument(
        "--max-steps",
        type=int,
        help="Smoke-only optimizer step override; output is marked non-submittable.",
    )
    train_qlora.add_argument(
        "--output-dir",
        help="Run-root-relative smoke output; the checked-in training path is unchanged.",
    )


def _phase1_model_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", default="data/raw/input")
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument(
        "--model-split-manifest",
        default=(
            "outputs/mining/model-datasets/phase1-manual-five-type-v1/"
            "split_manifest.json"
        ),
    )
    parser.add_argument(
        "--frozen-split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )


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
    registry_operations = registry.add_subparsers(dest="data_registry_command", required=True)
    registry_validate = registry_operations.add_parser(
        "validate", help="Validate registry v2 and print a source summary."
    )
    registry_validate.set_defaults(handler="data_registry_validate")
    registry_validate.add_argument("--registry", default="data/sources/mining_registry.yaml")
    registry_validate.add_argument(
        "--processing-index",
        help="Optional source-processing status file whose docs/config paths must exist.",
    )
    registry_validate.add_argument(
        "--repository-root",
        default=".",
        help="Repository root used to validate processing-index paths.",
    )

    artifact = operations.add_parser(
        "artifact",
        help="Restore immutable content-addressed source bytes for local processing.",
    )
    artifact_operations = artifact.add_subparsers(dest="data_artifact_command", required=True)
    artifact_materialize = artifact_operations.add_parser(
        "materialize",
        help="Atomically hydrate one CAS object to a verified local file.",
    )
    artifact_materialize.set_defaults(handler="data_artifact_materialize")
    artifact_materialize.add_argument("--store", required=True)
    artifact_materialize.add_argument("--sha256", required=True)
    artifact_materialize.add_argument("--output", required=True)
    artifact_materialize.add_argument("--expected-byte-size", type=int)

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
    dataset_operations = dataset.add_subparsers(dest="data_dataset_command", required=True)
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
    dataset_evidence = dataset_operations.add_parser(
        "attach-block-evidence",
        help="Attach source-section evidence tiers without changing annotation spans.",
    )
    dataset_evidence.set_defaults(handler="data_dataset_attach_block_evidence")
    dataset_evidence.add_argument("--documents", required=True)
    dataset_evidence.add_argument("--annotations", required=True)
    dataset_evidence.add_argument("--policy", required=True)
    dataset_evidence.add_argument("--output", required=True)
    dataset_evidence.add_argument("--report-output", required=True)
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
    dataset_reconcile.add_argument("--labeler-id", default="exact-duplicate-consensus:v1")
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
    dataset_harmonize.add_argument("--alias-overlay-source", action="append", default=[])
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
    dataset_exact_quote = dataset_operations.add_parser(
        "build-exact-quote-curriculum",
        help="Compile licensed train spans into source-label-preserving instructions.",
    )
    dataset_exact_quote.set_defaults(
        handler="data_dataset_build_exact_quote_curriculum"
    )
    dataset_exact_quote.add_argument(
        "--registry",
        default="data/sources/mining_registry.yaml",
    )
    dataset_exact_quote.add_argument("--source-id", required=True)
    dataset_exact_quote.add_argument("--spans", required=True)
    dataset_exact_quote.add_argument("--spans-manifest", required=True)
    dataset_exact_quote.add_argument("--label", action="append", required=True)
    dataset_exact_quote.add_argument("--output-dir", required=True)
    dataset_source_splits = dataset_operations.add_parser(
        "freeze-source-splits",
        help="Freeze source-declared train/validation/test assignments.",
    )
    dataset_source_splits.set_defaults(handler="data_dataset_freeze_source_splits")
    dataset_source_splits.add_argument("--documents", required=True)
    dataset_source_splits.add_argument("--metadata-key", default="source_split")
    dataset_source_splits.add_argument(
        "--map",
        action="append",
        required=True,
        help="Source and target split as SOURCE=TARGET.",
    )
    dataset_source_splits.add_argument("--output", required=True)

    lexicon = operations.add_parser(
        "lexicon", help="Build mined mention inventories for terminology review."
    )
    lexicon_operations = lexicon.add_subparsers(dest="data_lexicon_command", required=True)
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
    attach_links = lexicon_operations.add_parser(
        "attach-exact-links",
        help="Attach pinned exact-unique crosswalk evidence to mined annotations.",
    )
    attach_links.set_defaults(handler="data_lexicon_attach_exact_links")
    attach_links.add_argument("--annotations", required=True)
    attach_links.add_argument("--crosswalk", required=True)
    attach_links.add_argument("--policy", required=True)
    attach_links.add_argument("--output", required=True)
    attach_links.add_argument("--report-output", required=True)
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
    dailymed_product_aliases = lexicon_operations.add_parser(
        "propose-dailymed-product-aliases",
        help="Aggregate exact DailyMed product links within one frozen split.",
    )
    dailymed_product_aliases.set_defaults(handler="data_lexicon_propose_dailymed_product_aliases")
    dailymed_product_aliases.add_argument("--links", required=True)
    dailymed_product_aliases.add_argument("--split-manifest", required=True)
    dailymed_product_aliases.add_argument("--split", required=True)
    dailymed_product_aliases.add_argument("--output", required=True)
    dailymed_product_aliases.add_argument("--decisions-output", required=True)
    dailymed_product_aliases.add_argument("--report-output", required=True)

    mapping = operations.add_parser(
        "mapping", help="Compile source crosswalk releases into queryable indexes."
    )
    mapping_operations = mapping.add_subparsers(dest="data_mapping_command", required=True)
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

    mapping_ndc = mapping_operations.add_parser(
        "compile-rxnorm-ndc",
        help="Compile active package NDC attributes from a pinned RxNorm full release.",
    )
    mapping_ndc.set_defaults(handler="data_mapping_compile_rxnorm_ndc")
    mapping_ndc.add_argument("--source", required=True)
    mapping_ndc.add_argument("--source-version", required=True)
    mapping_ndc.add_argument("--expected-source-sha256", required=True)
    mapping_ndc.add_argument("--archive-member", default="rrf/RXNSAT.RRF")
    mapping_ndc.add_argument("--output", required=True)
    mapping_ndc.add_argument("--index-output", required=True)
    mapping_ndc.add_argument("--report-output", required=True)

    mapping_products = mapping_operations.add_parser(
        "link-dailymed-products",
        help="Link products only when exact SPL/version and RxNorm NDC evidence agree.",
    )
    mapping_products.set_defaults(handler="data_mapping_link_dailymed_products")
    mapping_products.add_argument("--documents", required=True)
    mapping_products.add_argument("--dailymed-mapping-index", required=True)
    mapping_products.add_argument("--rxnorm-ndc-index", required=True)
    mapping_products.add_argument("--links-output", required=True)
    mapping_products.add_argument("--decisions-output", required=True)
    mapping_products.add_argument("--report-output", required=True)

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
    ontology_hpo = ontology_operations.add_parser(
        "compile-hpo-associations",
        help="Preserve and aggregate HPO disease-phenotype and disease-gene evidence.",
    )
    ontology_hpo.set_defaults(handler="data_ontology_compile_hpo_associations")
    ontology_hpo.add_argument("--hpoa", required=True)
    ontology_hpo.add_argument("--genes", required=True)
    ontology_hpo.add_argument("--hpo-concepts", required=True)
    ontology_hpo.add_argument("--source-version", required=True)
    ontology_hpo.add_argument("--output-dir", required=True)

    knowledge = operations.add_parser(
        "knowledge",
        help="Promote mined evidence into strict runtime knowledge artifacts.",
    )
    knowledge_operations = knowledge.add_subparsers(
        dest="data_knowledge_command",
        required=True,
    )
    knowledge_abbreviations = knowledge_operations.add_parser(
        "mine-abbreviations",
        help="Mine explicit definitions from frozen splits and benchmark retrieval impact.",
    )
    knowledge_abbreviations.set_defaults(handler="data_knowledge_mine_abbreviations")
    knowledge_abbreviations.add_argument("--documents", required=True)
    knowledge_abbreviations.add_argument("--artifacts", required=True)
    knowledge_abbreviations.add_argument("--split-manifest", required=True)
    knowledge_abbreviations.add_argument("--policy", required=True)
    knowledge_abbreviations.add_argument(
        "--base-abbreviations",
        action="append",
        default=[],
        help="Existing abbreviation JSONL used to reject duplicate or conflicting definitions.",
    )
    knowledge_abbreviations.add_argument(
        "--index",
        help="Optional pinned terminology SQLite index for held-out retrieval evaluation.",
    )
    knowledge_abbreviations.add_argument(
        "--source",
        action="append",
        default=[],
        help="Canonical terminology JSONL used to validate --index; repeat as needed.",
    )
    knowledge_abbreviations.add_argument(
        "--alias-overlay",
        action="append",
        default=[],
        help="Alias overlay used to build --index; repeat as needed.",
    )
    knowledge_abbreviations.add_argument("--retrieval-limit", type=int, default=20)
    knowledge_abbreviations.add_argument("--definitions-output", required=True)
    knowledge_abbreviations.add_argument("--candidates-output", required=True)
    knowledge_abbreviations.add_argument("--table-output", required=True)
    knowledge_abbreviations.add_argument("--runtime-table-output", required=True)
    knowledge_abbreviations.add_argument("--conflicts-output", required=True)
    knowledge_abbreviations.add_argument("--benchmark-output", required=True)
    knowledge_abbreviations.add_argument("--report-output", required=True)
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
    knowledge_recognition.set_defaults(handler="data_knowledge_compile_recognition")
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
    recognition_benchmark.add_argument(
        "--require-document-metadata",
        action="append",
        default=[],
        help=(
            "Evaluate only documents carrying this metadata key. Repeat for multiple keys; "
            "use this when a source has representation-specific annotation coverage."
        ),
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
        "--preferred-code-system",
        action="append",
        default=[],
        metavar="ENTITY_TYPE=CODE_SYSTEM",
        help=(
            "Select a canonical endpoint from cross-system annotation links; repeat to "
            "allow multiple systems for one entity type."
        ),
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

    relation = operations.add_parser("relation", help="Run a local relation-labeler adapter.")
    relation_operations = relation.add_subparsers(dest="data_relation_command", required=True)
    relation_propose = relation_operations.add_parser(
        "propose", help="Generate provenance-bearing relation proposals."
    )
    relation_propose.set_defaults(handler="data_relation_propose")
    relation_propose.add_argument("--documents", required=True)
    relation_propose.add_argument("--annotations", required=True)
    relation_propose.add_argument(
        "--adapter", required=True, help="Local factory in module:attribute form."
    )
    relation_propose.add_argument("--adapter-config", help="YAML/JSON factory config mapping.")
    relation_propose.add_argument("--output", required=True)
    relation_cooccurrence = relation_operations.add_parser(
        "mine-cooccurrence",
        help=("Mine non-causal same-sentence evidence from a source-pinned training slice."),
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
    coverage_operations = coverage.add_subparsers(dest="data_coverage_command", required=True)
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
    snapshot_operations = snapshot.add_subparsers(dest="data_snapshot_command", required=True)
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
    snapshot_freeze.add_argument(
        "--dedup-mode",
        choices=("near", "exact"),
        default="near",
        help=(
            "Use exact for very large source-grouped corpora; near computes SimHash "
            "over every document."
        ),
    )
    snapshot_freeze.add_argument("--max-synthetic-fraction", type=float, default=0.4)
    snapshot_freeze.add_argument("--manifest-only", action="store_true")
    snapshot_freeze.add_argument("--skip-agreement-gate", action="store_true")

    release = operations.add_parser(
        "release",
        help="Lock or verify a portable mining-derived data/model release.",
    )
    release_operations = release.add_subparsers(dest="data_release_command", required=True)
    release_lock = release_operations.add_parser(
        "lock",
        help="Fingerprint declared datasets, knowledge, benchmarks, and code inputs.",
    )
    release_lock.set_defaults(handler="data_release_lock")
    release_lock.add_argument("--spec", required=True)
    release_lock.add_argument("--output", required=True)
    release_verify = release_operations.add_parser(
        "verify",
        help="Verify a lock after restoring or rebuilding artifacts on another machine.",
    )
    release_verify.set_defaults(handler="data_release_verify")
    release_verify.add_argument("--manifest", required=True)
    release_verify.add_argument(
        "--root",
        default=".",
        help="Local release root; it is never persisted in the portable lock.",
    )
    release_verify.add_argument(
        "--require-optional",
        action="store_true",
        help="Treat artifacts intentionally absent when locked as required during verification.",
    )
    release_verify.add_argument(
        "--store",
        help=(
            "Local directory, file:// URI, or fsspec URI containing external CAS objects. "
            "The URI is used only for this verification and is never persisted."
        ),
    )
    release_verify.add_argument(
        "--require-cas-objects",
        action="store_true",
        help="Fail when external source objects cannot be checked in the selected store.",
    )
    release_verify.add_argument(
        "--verify-cas-content",
        action="store_true",
        help="Stream and hash external CAS objects in addition to checking their existence.",
    )

    run = operations.add_parser("run", help="Run a resumable declarative mining plan.")
    run.set_defaults(handler="data_run")
    run.add_argument("--plan", required=True)
