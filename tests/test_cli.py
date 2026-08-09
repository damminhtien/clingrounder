"""Consolidated CLI command parity and profile behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clingrounder.cli import main
from clingrounder.cli.parser import build_parser
from clingrounder.utils.io import read_jsonl, write_jsonl


def test_qwen_final_supervision_command_has_governed_defaults() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "qwen",
            "propose-final-supervision",
            "--config",
            "configs/qwen.yaml",
            "--output-dir",
            "outputs/qwen-source",
        ]
    )

    assert args.handler == "benchmark_phase1_qwen_final_supervision_propose"
    assert args.training_governance.endswith("phase1-training-governance-2026-07-30.yaml")
    assert args.extraction_mode == "recall_and_targeted"


def test_pipeline_run_requires_an_explicit_profile() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "pipeline",
                "run",
                "--input",
                "data/samples/sample_notes.jsonl",
                "--output",
                "outputs/predictions.jsonl",
            ]
        )


def test_public_benchmark_suite_parser_accepts_named_configs() -> None:
    args = build_parser("benchmark").parse_args(
        [
            "benchmark",
            "suite",
            "--benchmark",
            "benchmarks/vi_clinical_grounding_v1",
            "--config",
            "exact=configs/benchmarks/vi_clinical_grounding_v1/exact.yaml",
            "--config",
            "full=configs/benchmarks/vi_clinical_grounding_v1/full.yaml",
            "--output",
            "/tmp/benchmark-suite",
        ]
    )

    assert args.handler == "benchmark_dataset_suite"
    assert args.config == [
        "exact=configs/benchmarks/vi_clinical_grounding_v1/exact.yaml",
        "full=configs/benchmarks/vi_clinical_grounding_v1/full.yaml",
    ]


def test_public_benchmark_review_pack_parser_accepts_assignments() -> None:
    args = build_parser("benchmark").parse_args(
        [
            "benchmark",
            "review-pack",
            "--benchmark",
            "benchmarks/vi_clinical_grounding_v1",
            "--output",
            "/tmp/review-pack",
            "--reviewer",
            "alice",
            "--reviewer",
            "bob",
            "--double-review-fraction",
            "0.25",
            "--seed",
            "7",
        ]
    )

    assert args.handler == "benchmark_review_pack"
    assert args.reviewer == ["alice", "bob"]
    assert args.double_review_fraction == 0.25
    assert args.seed == 7


def test_review_quality_parser_can_emit_public_agreement_artifact() -> None:
    args = build_parser("research").parse_args(
        [
            "data",
            "review",
            "quality",
            "--documents",
            "documents.jsonl",
            "--proposals",
            "proposals.jsonl",
            "--output",
            "quality.json",
            "--dataset-id",
            "vi-clinical-grounding-v1",
            "--dataset-version",
            "1.0.0",
            "--benchmark-output",
            "agreement.json",
        ]
    )

    assert args.handler == "data_review_quality"
    assert args.benchmark_output == "agreement.json"


def test_pipeline_config_inspection_exposes_effective_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "pipeline",
                "inspect-config",
                "--config",
                "configs/pipeline/clinical-baseline.yaml",
                "--check-resources",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["profile"]["id"] == "clinical-baseline"
    assert report["effective_config"]["pipeline"]["compiled_options"]["max_candidates"] == 20
    assert all(resource["exists"] for resource in report["resources"])


def test_pipeline_list_profiles_reports_support_status(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["pipeline", "list-profiles", "--check-resources"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["errors"] == []
    profiles = {item["profile"]["id"]: item for item in report["profiles"]}
    assert profiles["clinical-baseline"]["profile"]["portability"] == "portable"
    assert profiles["full-terminology"]["profile"]["portability"] == "local"


def test_pipeline_run_writes_reproducibility_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "predictions.jsonl"
    assert (
        main(
            [
                "pipeline",
                "run",
                "--config",
                "configs/pipeline/clinical-baseline.yaml",
                "--input",
                "data/samples/sample_notes.jsonl",
                "--output",
                str(output),
                "--parallel-backend",
                "serial",
            ]
        )
        == 0
    )

    command_report = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(command_report["manifest"]).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "clingrounder.pipeline-run.v1"
    assert manifest["profile"]["profile"]["id"] == "clinical-baseline"
    assert manifest["input"]["sha256"]
    assert manifest["output"]["sha256"]


def test_joint_span_final_fit_command_requires_independent_source_roles() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "joint-span",
            "prepare-final-fit",
            "--model-source",
            "qwen=outputs/qwen-source",
            "--source-role",
            "qwen=llm",
            "--output-dir",
            "outputs/joint-span",
        ]
    )

    assert args.handler == "benchmark_phase1_joint_span_prepare_final_fit"
    assert args.dictionary[0].endswith("phase1_seed_tt06_rxnorm_controlled_concepts.jsonl")


def test_joint_span_token_bundle_command_accepts_rule_medication_bootstrap() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "joint-span",
            "prepare-token-bundle",
            "--dataset",
            "outputs/bundle/spans.jsonl",
            "--dataset-manifest",
            "outputs/bundle/manifest.json",
            "--output-dir",
            "outputs/joint-span-bundle",
        ]
    )

    assert args.handler == "benchmark_phase1_joint_span_prepare_token_bundle"
    assert args.model_source == []


def test_joint_span_train_command_exposes_pinned_model_and_dataset_inputs() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "joint-span",
            "train",
            "--dataset",
            "outputs/joint/examples.jsonl",
            "--dataset-manifest",
            "outputs/joint/manifest.json",
            "--output-dir",
            "outputs/model",
            "--model-id",
            "FacebookAI/xlm-roberta-base",
            "--revision",
            "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
            "--bf16",
        ]
    )

    assert args.handler == "benchmark_phase1_joint_span_train"
    assert args.bf16 is True
    assert args.max_length == 384


def test_joint_span_token_source_command_exposes_checkpoint_provenance() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "joint-span",
            "materialize-token-source",
            "--model-path",
            "outputs/models/final-model",
            "--model-fingerprint",
            "a" * 64,
            "--model-id",
            "FacebookAI/xlm-roberta-base",
            "--base-revision",
            "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
            "--output-dir",
            "outputs/xlmr-source",
            "--device",
            "cuda",
        ]
    )

    assert args.handler == "benchmark_phase1_joint_span_materialize_token_source"
    assert args.device == "cuda"
    assert args.source_name == "xlmr"


def test_joint_span_token_bundle_source_command_uses_pinned_bundle_inputs() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "joint-span",
            "materialize-token-bundle-source",
            "--dataset",
            "outputs/bundle/spans.jsonl",
            "--dataset-manifest",
            "outputs/bundle/manifest.json",
            "--model-path",
            "outputs/model",
            "--model-fingerprint",
            "a" * 64,
            "--model-id",
            "FacebookAI/xlm-roberta-base",
            "--base-revision",
            "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
            "--output-dir",
            "outputs/xlmr-bundle-source",
        ]
    )

    assert args.handler == "benchmark_phase1_joint_span_materialize_token_bundle_source"
    assert args.device == "cpu"


def test_qwen_token_bundle_command_has_two_pass_default() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "qwen",
            "propose-token-bundle",
            "--config",
            "configs/qwen.yaml",
            "--dataset",
            "outputs/bundle/spans.jsonl",
            "--dataset-manifest",
            "outputs/bundle/manifest.json",
            "--output-dir",
            "outputs/qwen-bundle-source",
        ]
    )

    assert args.handler == "benchmark_phase1_qwen_token_bundle_propose"
    assert args.extraction_mode == "recall_and_targeted"


def test_joint_span_run_command_accepts_only_a_pinned_config() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "joint-span",
            "run",
            "--config",
            "configs/phase1_joint_span_submission.yaml",
        ]
    )

    assert args.handler == "benchmark_phase1_joint_span_run"


def test_joint_span_calibration_command_requires_oof_provenance() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "joint-span",
            "calibrate",
            "--observations",
            "outputs/joint/oof-observations.jsonl",
            "--training-family-fingerprint",
            "a" * 64,
            "--fold-assignment-sha256",
            "b" * 64,
            "--output",
            "outputs/joint/calibration.json",
        ]
    )

    assert args.handler == "benchmark_phase1_joint_span_calibrate"
    assert args.false_positive_cost == 1.0


def test_terminology_build_and_inspect_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "concepts.jsonl"
    source.write_text(
        '{"concept_id":"D1","code":"I10","code_system":"ICD-10",'
        '"canonical_name":"tăng huyết áp","semantic_type":"DISEASE"}\n',
        encoding="utf-8",
    )
    index = tmp_path / "terminology.sqlite3"
    manifest = tmp_path / "manifest.json"

    assert (
        main(
            [
                "terminology",
                "build",
                "--source",
                str(source),
                "--output",
                str(index),
                "--manifest-output",
                str(manifest),
            ]
        )
        == 0
    )
    build_payload = json.loads(capsys.readouterr().out)
    assert build_payload["concept_count"] == 1
    assert json.loads(manifest.read_text(encoding="utf-8")) == build_payload

    assert (
        main(
            [
                "terminology",
                "inspect",
                "--index",
                str(index),
                "--source",
                str(source),
                "--query",
                "tăng huyết áp",
                "--entity-type",
                "DISEASE",
            ]
        )
        == 0
    )
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["results"][0]["code"] == "I10"


def test_terminology_query_set_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    overlay = tmp_path / "aliases.jsonl"
    queries = tmp_path / "queries.jsonl"
    manifest = tmp_path / "query_manifest.json"
    write_jsonl(
        overlay,
        [
            {
                "alias": "cao huyết áp",
                "code": "I10",
                "code_system": "ICD-10",
                "semantic_type": "DISEASE",
            }
        ],
    )

    assert (
        main(
            [
                "terminology",
                "query-set",
                "--alias-overlay",
                str(overlay),
                "--output",
                str(queries),
                "--manifest-output",
                str(manifest),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["query_count"] == 1
    assert read_jsonl(queries)[0]["expected_codes"] == ["I10"]
    assert json.loads(manifest.read_text(encoding="utf-8")) == payload


def test_terminology_query_set_command_supports_heldout_linked_proposals(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    proposals = tmp_path / "proposals.jsonl"
    queries = tmp_path / "queries.jsonl"
    manifest = tmp_path / "query_manifest.json"
    write_jsonl(
        proposals,
        [
            {
                "normalized_alias": "cao huyết áp",
                "code": "I10",
                "code_system": "ICD-10",
                "semantic_type": "DISEASE",
                "surface_variants": [{"surface": "Cao huyết áp"}],
            }
        ],
    )

    assert (
        main(
            [
                "terminology",
                "query-set",
                "--linked-proposal",
                str(proposals),
                "--output",
                str(queries),
                "--manifest-output",
                str(manifest),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["slice_counts"] == {
        "alias_unseen_in_reference": 1,
        "code_unseen_in_reference": 1,
    }


def test_terminology_benchmark_prints_summary_and_saves_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "concepts.jsonl"
    queries = tmp_path / "queries.jsonl"
    index = tmp_path / "terminology.sqlite3"
    report = tmp_path / "report.json"
    write_jsonl(
        source,
        [
            {
                "concept_id": "D1",
                "code": "I10",
                "code_system": "ICD-10",
                "canonical_name": "tăng huyết áp",
                "semantic_type": "DISEASE",
            }
        ],
    )
    write_jsonl(
        queries,
        [
            {
                "query_id": "missing-1",
                "mention": "không tồn tại",
                "entity_type": "DISEASE",
                "code_system": "ICD-10",
                "expected_codes": ["I10"],
            }
        ],
    )
    assert main(["terminology", "build", "--source", str(source), "--output", str(index)]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "terminology",
                "benchmark",
                "--index",
                str(index),
                "--source",
                str(source),
                "--queries",
                str(queries),
                "--output",
                str(report),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    detailed = json.loads(report.read_text(encoding="utf-8"))
    assert "errors" not in summary["modes"]["exact"]
    assert detailed["modes"]["exact"]["errors"][0]["query_id"] == "missing-1"


def test_validate_command_profiles_hash_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = read_jsonl("data/samples/gold.jsonl")
    rows[0]["text_hash"] = "stale"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, rows)
    common = [
        "validate",
        "--pred",
        str(predictions),
        "--documents",
        "data/samples/sample_notes.jsonl",
        "--dictionary",
        "data/dictionaries/seed_concepts.jsonl",
    ]

    assert main([*common, "--profile", "development"]) == 0
    development = json.loads(capsys.readouterr().out)
    assert development["warnings"] >= 1

    assert main([*common, "--profile", "release"]) == 1
    release = json.loads(capsys.readouterr().out)
    assert release["errors"] >= 1


def test_release_validation_requires_terminology_for_assigned_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, read_jsonl("data/samples/gold.jsonl"))

    exit_code = main(
        [
            "validate",
            "--pred",
            str(predictions),
            "--documents",
            "data/samples/sample_notes.jsonl",
            "--profile",
            "release",
        ]
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code == 1
    assert report["errors"] > 0
    assert "terminology_membership_unavailable" in captured.err


def test_evaluate_command_writes_error_analysis(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    error_path = tmp_path / "errors.csv"

    assert (
        main(
            [
                "evaluate",
                "--gold",
                "data/samples/gold.jsonl",
                "--pred",
                "data/samples/gold.jsonl",
                "--error-analysis",
                str(error_path),
            ]
        )
        == 0
    )
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["span_exact"]["f1"] == 1.0
    assert error_path.exists()


@pytest.mark.integration
def test_pipeline_run_command_writes_predictions(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"

    assert (
        main(
            [
                "pipeline",
                "run",
                "--config",
                "configs/pipeline/clinical-baseline.yaml",
                "--input",
                "data/samples/sample_notes.jsonl",
                "--output",
                str(output),
                "--parallel-backend",
                "serial",
            ]
        )
        == 0
    )
    assert len(read_jsonl(output)) == 1


@pytest.mark.release
def test_phase1_benchmark_command_builds_strict_zip(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("Tăng huyết áp", encoding="utf-8")
    output_dir = tmp_path / "output"
    archive = tmp_path / "submission.zip"

    assert (
        main(
            [
                "benchmark",
                "phase1",
                "submission",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--zip",
                str(archive),
                "--parallel-backend",
                "serial",
            ]
        )
        == 0
    )
    assert archive.exists()
