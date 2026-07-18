"""Consolidated CLI command parity and profile behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.cli import main
from medical_kg_nlp.utils.io import read_jsonl, write_jsonl


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
