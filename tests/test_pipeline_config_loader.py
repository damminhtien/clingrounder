"""Portable loading contracts for reusable pipeline profiles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from medical_kg_nlp.benchmarks.phase1.runner import (
    Phase1BenchmarkConfig,
    run_phase1_benchmark,
)
from medical_kg_nlp.pipeline import PipelineFactory, ResolvedPipelineConfig
from medical_kg_nlp.schema.types import CodeSystem
from medical_kg_nlp.terminology import build_terminology_index


def test_pipeline_profile_paths_do_not_depend_on_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = tmp_path / "project" / "configs" / "pipeline"
    profile_dir.mkdir(parents=True)
    profile = profile_dir / "profile.yaml"
    profile.write_text(
        """\
terminology:
  recognition_path: ../../data/recognition.jsonl
  normalization_paths:
    - ../../data/icd.jsonl
    - ../../data/rxnorm.jsonl
  normalization_index_path: ../../cache/terminology.sqlite3
  normalization_alias_overlay_paths:
    - ../../data/aliases.jsonl
  knowledge_graph_index_path: ../../cache/knowledge.sqlite3
  cache_dir: ../../cache
  reviewed_mention_path: ../../data/reviewed.jsonl
  additional_recognition_path: ../../data/vn.jsonl
  additional_recognition_paths:
    - ../../data/mined.jsonl
  abbreviation_path: ../../data/abbreviations.jsonl
  alias_overlay_path: ../../data/vietnamese_aliases.jsonl
pipeline:
  enable_relations: false
  enable_relation_kg_validation: false
models:
  entity_extractor:
    run_spec: ../../models/run.yaml
    model_id: ../../models/final-model
    revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  candidate_reranker:
    model_id: medical-org/reranker
    revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
""",
        encoding="utf-8",
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    resolved = ResolvedPipelineConfig.load(profile)

    assert resolved.factory_config.recognition_dictionary_path == str(
        tmp_path / "project" / "data" / "recognition.jsonl"
    )
    assert resolved.factory_config.normalization_dictionary_paths == (
        str(tmp_path / "project" / "data" / "icd.jsonl"),
        str(tmp_path / "project" / "data" / "rxnorm.jsonl"),
    )
    assert resolved.factory_config.models.entity_extractor is not None
    assert resolved.factory_config.models.entity_extractor.model_id == str(
        tmp_path / "project" / "models" / "final-model"
    )
    assert resolved.factory_config.models.candidate_reranker is not None
    assert (
        resolved.factory_config.models.candidate_reranker.model_id
        == "medical-org/reranker"
    )


def test_rebased_pipeline_profile_round_trips_effective_paths(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "pipeline.yaml"
    source.write_text(
        """\
terminology:
  recognition_path: data/recognition.jsonl
  abbreviation_path: data/abbreviations.jsonl
  alias_overlay_path: null
pipeline:
  enable_relations: false
  enable_relation_kg_validation: false
models:
  entity_extractor:
    run_spec: models/run.yaml
    model_id: models/final-model
    revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
""",
        encoding="utf-8",
    )
    resolved = ResolvedPipelineConfig.load(source)
    destination = tmp_path / "artifacts" / "run-1" / "pipeline.yaml"
    destination.parent.mkdir(parents=True)
    destination.write_text(
        yaml.safe_dump(
            resolved.payload_for(destination),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    reloaded = ResolvedPipelineConfig.load(destination)
    written = yaml.safe_load(destination.read_text(encoding="utf-8"))

    assert reloaded.factory_config == resolved.factory_config
    assert not Path(written["terminology"]["recognition_path"]).is_absolute()
    assert not Path(
        written["models"]["entity_extractor"]["run_spec"]
    ).is_absolute()
    assert not Path(
        written["models"]["entity_extractor"]["model_id"]
    ).is_absolute()


def test_resolved_profile_composes_and_builds_zip_outside_profile_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    data_dir = project / "data"
    profile_dir = project / "configs" / "pipeline"
    cache_dir = project / "cache"
    input_dir = project / "input"
    for directory in (data_dir, profile_dir, cache_dir, input_dir):
        directory.mkdir(parents=True)

    recognition = data_dir / "recognition.jsonl"
    normalization = data_dir / "normalization.jsonl"
    abbreviations = data_dir / "abbreviations.jsonl"
    recognition.write_text(_concept_rows(), encoding="utf-8")
    normalization.write_text(_concept_rows(), encoding="utf-8")
    abbreviations.write_text("", encoding="utf-8")
    index_path = cache_dir / "terminology.sqlite3"
    build_terminology_index((normalization,), output_path=index_path)
    (input_dir / "1.txt").write_text("Đái tháo đường", encoding="utf-8")

    profile = profile_dir / "profile.yaml"
    profile.write_text(
        """\
terminology:
  recognition_path: ../../data/recognition.jsonl
  normalization_paths:
    - ../../data/normalization.jsonl
  normalization_index_path: ../../cache/terminology.sqlite3
  abbreviation_path: ../../data/abbreviations.jsonl
  alias_overlay_path: null
  reviewed_mention_path: null
pipeline:
  candidate_sources: [exact]
  enable_context: false
  enable_relations: false
  enable_relation_kg_validation: false
""",
        encoding="utf-8",
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    resolved = ResolvedPipelineConfig.load(profile)
    runner = PipelineFactory.from_config(resolved.factory_config)
    repository = runner.components.terminology_repository
    assert repository is not None
    assert repository.get_by_code(CodeSystem.ICD10, "E11.9") is not None
    assert repository.get_by_code(CodeSystem.RXNORM, "6809") is not None

    output_dir = project / "submission"
    zip_path = project / "submission.zip"
    report = run_phase1_benchmark(
        Phase1BenchmarkConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            zip_path=zip_path,
            dictionary_path=recognition,
            abbreviation_path=abbreviations,
            pipeline_config_path=profile,
            validation_dictionary_paths=(normalization,),
            backend="serial",
            workers=1,
        )
    )

    assert report["validation_issues"] == 0
    assert zip_path.is_file()


def test_checked_in_full_profile_composes_tt06_and_rxnorm_without_btc_memory() -> None:
    resolved = ResolvedPipelineConfig.load("configs/phase1_full.yaml")
    sources = tuple(Path(path) for path in resolved.factory_config.normalization_dictionary_paths)

    assert [path.name for path in sources] == [
        "tt06_icd10_concepts.jsonl",
        "rxnorm_full_07062026_concepts.jsonl",
    ]
    assert resolved.factory_config.reviewed_mention_path is None
    assert Path(resolved.factory_config.recognition_dictionary_path).name == (
        "phase1_seed_tt06_rxnorm_controlled_concepts.jsonl"
    )


def test_round2_full_profile_adds_reviewed_recognition_sources() -> None:
    resolved = ResolvedPipelineConfig.load("configs/phase1_round2_full.yaml")
    factory = resolved.factory_config

    assert [
        Path(path).name for path in factory.additional_recognition_dictionary_paths
    ] == ["vn_clinical_lexicon_concepts.jsonl", "recognition_concepts.jsonl"]
    assert factory.reviewed_mention_path is None
    assert factory.options.candidate_sources == ("exact",)
    assert factory.options.link_max_qualified_candidates == 1


def _concept_rows() -> str:
    rows = (
        {
            "concept_id": "ICD:E11.9",
            "code": "E11.9",
            "code_system": "ICD-10",
            "canonical_name": "Đái tháo đường",
            "semantic_type": "DISEASE",
            "source": "test",
        },
        {
            "concept_id": "RX:6809",
            "code": "6809",
            "code_system": "RxNorm",
            "canonical_name": "metformin",
            "semantic_type": "DRUG",
            "rxnorm_tty": "IN",
            "ingredient": "metformin",
            "source": "test",
        },
    )
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
