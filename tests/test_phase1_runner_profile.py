from __future__ import annotations

from pathlib import Path

from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.benchmarks.phase1.runner import (
    Phase1BenchmarkConfig,
    build_phase1_factory_config,
    _validation_paths,
)


def test_phase1_profile_preserves_mined_sources_and_disables_relations(tmp_path: Path) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        """
terminology:
  recognition_path: profile-seed.jsonl
  additional_recognition_paths:
    - mined-a.jsonl
    - mined-b.jsonl
  normalization_paths:
    - icd.jsonl
  normalization_index_path: terminology.sqlite3
pipeline:
  enable_relations: true
  enable_relation_kg_validation: true
  candidate_sources: [exact, bm25]
""",
        encoding="utf-8",
    )
    config = build_phase1_factory_config(
        Phase1BenchmarkConfig(
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
            zip_path=tmp_path / "output.zip",
            dictionary_path=tmp_path / "seed.jsonl",
            abbreviation_path=tmp_path / "abbr.jsonl",
            pipeline_config_path=profile,
        )
    )

    assert config.recognition_dictionary_path == str(tmp_path / "seed.jsonl")
    assert config.abbreviation_path == str(tmp_path / "abbr.jsonl")
    assert config.additional_recognition_dictionary_paths == ("mined-a.jsonl", "mined-b.jsonl")
    assert config.normalization_dictionary_paths == ("icd.jsonl",)
    assert config.options.candidate_sources == ("exact", "bm25")
    assert config.options.enable_relations is False
    assert config.options.enable_relation_kg_validation is False


def test_validation_dictionary_paths_are_explicit_and_ordered(tmp_path: Path) -> None:
    config = Phase1BenchmarkConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        zip_path=tmp_path / "output.zip",
        dictionary_path=tmp_path / "recognition.jsonl",
        abbreviation_path=tmp_path / "abbr.jsonl",
        validation_dictionary_paths=(tmp_path / "icd.jsonl", tmp_path / "rxnorm.jsonl"),
    )

    # The field is intentionally separate from dictionary_path: recognition can stay compact
    # while release validation covers every code system emitted by normalization. The runner
    # always includes the recognition source first.
    assert config.validation_dictionary_paths == (
        tmp_path / "icd.jsonl",
        tmp_path / "rxnorm.jsonl",
    )
    assert _validation_paths(config) == (
        tmp_path / "recognition.jsonl",
        tmp_path / "icd.jsonl",
        tmp_path / "rxnorm.jsonl",
    )


def test_submission_cli_accepts_private_manifest_and_hashed_output() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "submission",
            "--documents",
            "round2/documents.jsonl",
            "--source-archive-sha256",
            "a" * 64,
            "--run-root",
            "outputs/phase1/round2",
            "--provenance-input",
            "outputs/models/run_manifest.json",
        ]
    )

    assert args.documents == "round2/documents.jsonl"
    assert args.input_dir is None
    assert args.output_dir == "output"
    assert args.zip == "output.zip"
    assert args.provenance_input == ["outputs/models/run_manifest.json"]
