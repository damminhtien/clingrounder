"""Command-line surface owned by the optional Phase 1 benchmark plugin."""

from __future__ import annotations

import argparse

__all__ = ["register_phase1_cli"]


def register_phase1_cli(
    plugins: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Compatibility parser registration owned by the optional Phase 1 plugin."""

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
        "--assertion-overlay",
        help="Optional Phase 1 assertion-overlay JSONL selected explicitly by the benchmark.",
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
        default="configs/benchmarks/phase1/models/phase1-training-governance-2026-07-30.yaml",
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

    model_data_final_bundle = model_data_operations.add_parser(
        "build-final-fit-bundle",
        help="Combine final supervision with bounded Q&A/educational synthetic records.",
    )
    model_data_final_bundle.set_defaults(
        handler="benchmark_phase1_model_data_build_final_fit_bundle"
    )
    model_data_final_bundle.add_argument(
        "--final-dataset",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-final-supervision-five-type-v1/spans.jsonl"
        ),
    )
    model_data_final_bundle.add_argument(
        "--final-manifest",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-final-supervision-five-type-v1/manifest.json"
        ),
    )
    model_data_final_bundle.add_argument(
        "--augmentation-dataset",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-qa-edu-v1/spans.jsonl"
        ),
    )
    model_data_final_bundle.add_argument(
        "--augmentation-manifest",
        default=(
            "outputs/mining/model-datasets/"
            "phase1-manual-five-type-qa-edu-v1/manifest.json"
        ),
    )
    model_data_final_bundle.add_argument("--output-dir", required=True)
    model_data_final_bundle.add_argument(
        "--maximum-synthetic-fraction",
        type=float,
        default=0.4,
        help="Hard cap for approved Q&A/educational synthetic chunk share.",
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
        default="configs/benchmarks/phase1/pipeline/phase1-five-type-model-only.yaml",
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
        default="configs/benchmarks/phase1/models/phase1-training-governance-2026-07-30.yaml",
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
    qwen_token_bundle = qwen_operations.add_parser(
        "propose-token-bundle",
        help="Run Qwen exact-quote proposal passes over a governed mixed-genre token bundle.",
    )
    qwen_token_bundle.set_defaults(handler="benchmark_phase1_qwen_token_bundle_propose")
    qwen_token_bundle.add_argument("--config", required=True)
    qwen_token_bundle.add_argument("--dataset", required=True)
    qwen_token_bundle.add_argument("--dataset-manifest", required=True)
    qwen_token_bundle.add_argument("--bundle-build-manifest")
    qwen_token_bundle.add_argument("--output-dir", required=True)
    qwen_token_bundle.add_argument(
        "--extraction-mode",
        choices=("recall_only", "recall_and_targeted"),
        default="recall_and_targeted",
    )
    qwen_token_bundle.add_argument("--resume", action="store_true")

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
        default="configs/benchmarks/phase1/models/phase1-training-governance-2026-07-30.yaml",
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
    joint_span_prepare_bundle = joint_span_operations.add_parser(
        "prepare-token-bundle",
        help=(
            "Build a grouped mixed-genre lattice from the pinned final token bundle; model "
            "sources are optional for the initial rule/medication bootstrap."
        ),
    )
    joint_span_prepare_bundle.set_defaults(
        handler="benchmark_phase1_joint_span_prepare_token_bundle"
    )
    joint_span_prepare_bundle.add_argument("--dataset", required=True)
    joint_span_prepare_bundle.add_argument("--dataset-manifest", required=True)
    joint_span_prepare_bundle.add_argument("--bundle-build-manifest")
    joint_span_prepare_bundle.add_argument("--output-dir", required=True)
    joint_span_prepare_bundle.add_argument(
        "--model-source",
        action="append",
        default=[],
        help="Pinned source artifact as NAME=DIRECTORY; repeat for Qwen/XLM-R.",
    )
    joint_span_prepare_bundle.add_argument(
        "--source-role",
        action="append",
        default=[],
        help="Role for every model source as NAME=llm|token_model|ensemble.",
    )
    joint_span_prepare_bundle.add_argument(
        "--dictionary",
        action="append",
        default=[
            "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl",
            "data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl",
            "data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl",
        ],
        help="Canonical recognition dictionaries; repeat to add a pinned source.",
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
        default="configs/benchmarks/phase1/models/phase1-training-governance-2026-07-30.yaml",
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
    joint_span_token_bundle_source = joint_span_operations.add_parser(
        "materialize-token-bundle-source",
        help=(
            "Project one pinned local five-type checkpoint over the child texts used by mixed-genre "
            "joint-span OOF."
        ),
    )
    joint_span_token_bundle_source.set_defaults(
        handler="benchmark_phase1_joint_span_materialize_token_bundle_source"
    )
    joint_span_token_bundle_source.add_argument("--dataset", required=True)
    joint_span_token_bundle_source.add_argument("--dataset-manifest", required=True)
    joint_span_token_bundle_source.add_argument("--bundle-build-manifest")
    joint_span_token_bundle_source.add_argument("--model-path", required=True)
    joint_span_token_bundle_source.add_argument("--model-fingerprint", required=True)
    joint_span_token_bundle_source.add_argument("--model-id", required=True)
    joint_span_token_bundle_source.add_argument("--base-revision", required=True)
    joint_span_token_bundle_source.add_argument("--output-dir", required=True)
    joint_span_token_bundle_source.add_argument("--source-name", default="xlmr")
    joint_span_token_bundle_source.add_argument("--device", default="cpu")
    joint_span_token_bundle_source.add_argument("--batch-size", type=int, default=16)
    joint_span_token_bundle_source.add_argument("--max-length", type=int, default=512)
    joint_span_token_bundle_source.add_argument("--stride", type=int, default=64)
    joint_span_token_bundle_source.add_argument(
        "--default-confidence-threshold",
        type=float,
        default=0.0,
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
    joint_span_train.add_argument(
        "--training-family-dataset-sha256",
        help=(
            "Full supervised lattice SHA-256 shared by OOF folds and the final fit; "
            "required when --dataset is an OOF fold subset."
        ),
    )
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
    joint_span_train_oof = joint_span_operations.add_parser(
        "train-oof",
        help=(
            "Document-grouped cross-fit local joint-span verifiers and write calibration-ready "
            "OOF probabilities; no final-fit model is used for calibration."
        ),
    )
    joint_span_train_oof.set_defaults(handler="benchmark_phase1_joint_span_train_oof")
    joint_span_train_oof.add_argument("--dataset", required=True)
    joint_span_train_oof.add_argument("--dataset-manifest", required=True)
    joint_span_train_oof.add_argument("--output-dir", required=True)
    joint_span_train_oof.add_argument("--model-id", required=True)
    joint_span_train_oof.add_argument("--revision", required=True)
    joint_span_train_oof.add_argument("--initialization-model")
    joint_span_train_oof.add_argument("--initialization-fingerprint")
    joint_span_train_oof.add_argument("--fold-count", type=int, default=5)
    joint_span_train_oof.add_argument("--inference-device", default="cuda")
    joint_span_train_oof.add_argument("--max-length", type=int, default=384)
    joint_span_train_oof.add_argument("--train-batch-size", type=int, default=8)
    joint_span_train_oof.add_argument("--evaluation-batch-size", type=int, default=16)
    joint_span_train_oof.add_argument("--epochs", type=float, default=4.0)
    joint_span_train_oof.add_argument("--learning-rate", type=float, default=2e-5)
    joint_span_train_oof.add_argument("--weight-decay", type=float, default=0.01)
    joint_span_train_oof.add_argument("--warmup-ratio", type=float, default=0.08)
    joint_span_train_oof.add_argument("--gradient-accumulation-steps", type=int, default=1)
    joint_span_train_oof.add_argument("--seed", type=int, default=42)
    oof_precision = joint_span_train_oof.add_mutually_exclusive_group()
    oof_precision.add_argument("--fp16", action="store_true")
    oof_precision.add_argument("--bf16", action="store_true")
    joint_span_train_oof.add_argument("--use-cpu", action="store_true")
    joint_span_train_oof.add_argument("--cache-dir")
    joint_span_train_oof.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only completed folds whose manifest, model identity, and OOF rows validate.",
    )
    joint_span_calibrate = joint_span_operations.add_parser(
        "calibrate",
        help="Fit pinned genre/type calibration from document-grouped OOF joint verifier scores.",
    )
    joint_span_calibrate.set_defaults(handler="benchmark_phase1_joint_span_calibrate")
    joint_span_calibrate.add_argument("--observations", required=True)
    joint_span_calibrate.add_argument("--training-family-fingerprint", required=True)
    joint_span_calibrate.add_argument("--fold-assignment-sha256", required=True)
    joint_span_calibrate.add_argument("--output", required=True)
    joint_span_calibrate.add_argument("--report")
    joint_span_calibrate.add_argument("--false-positive-cost", type=float, default=1.0)
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

