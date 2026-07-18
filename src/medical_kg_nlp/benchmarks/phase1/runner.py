"""Reproducible Phase 1 benchmark build service used by the consolidated CLI."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import yaml

from medical_kg_nlp.benchmarks.phase1.phase1 import (
    load_phase1_text_documents,
    validate_phase1_submission_dir,
    validate_phase1_submission_zip,
    write_phase1_output_dir,
    zip_phase1_output_dir,
)
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.pipeline.factory import PipelineFactoryConfig
from medical_kg_nlp.pipeline.parallel_batch import (
    ParallelBatchOptions,
    ParallelBackend,
    run_batch_with_trace_parallel,
)

__all__ = [
    "BenchmarkExportPolicy",
    "Phase1BenchmarkConfig",
    "build_phase1_factory_config",
    "run_phase1_benchmark",
]

BenchmarkExportPolicy = Literal["empty", "pipeline"]


@dataclass(frozen=True)
class Phase1BenchmarkConfig:
    """Inputs and conservative export policies for one benchmark artifact."""

    input_dir: Path
    output_dir: Path
    zip_path: Path
    dictionary_path: Path
    abbreviation_path: Path
    pipeline_config_path: Path | None = None
    assertion_policy: BenchmarkExportPolicy = "pipeline"
    candidate_policy: BenchmarkExportPolicy = "pipeline"
    max_candidates: int = 5
    backend: ParallelBackend = "process"
    workers: int = 1
    chunksize: int = 4


def run_phase1_benchmark(config: Phase1BenchmarkConfig) -> dict[str, Any]:
    """Run, release-validate, and deterministically archive a Phase 1 submission."""

    documents = load_phase1_text_documents(config.input_dir)
    results = run_batch_with_trace_parallel(
        documents,
        factory_config=build_phase1_factory_config(config),
        parallel_options=ParallelBatchOptions(
            backend=config.backend,
            max_workers=config.workers,
            chunksize=config.chunksize,
        ),
    )
    predictions = [result.prediction for result in results]
    source_text_by_document = {
        document.document_id: document.text for document in documents
    }
    write_phase1_output_dir(
        predictions,
        config.output_dir,
        max_candidates=config.max_candidates,
        source_text_by_document=source_text_by_document,
        assertion_policy=config.assertion_policy,
        candidate_policy=config.candidate_policy,
    )
    dictionary = DictionaryStore.from_jsonl(config.dictionary_path)
    directory_issues = validate_phase1_submission_dir(
        config.input_dir,
        config.output_dir,
        dictionary=dictionary,
    )
    if directory_issues:
        raise ValueError(
            f"Phase 1 release validation failed with {len(directory_issues)} issue(s)"
        )
    zip_phase1_output_dir(config.output_dir, config.zip_path)
    zip_issues = validate_phase1_submission_zip(
        config.zip_path,
        input_dir=config.input_dir,
        dictionary=dictionary,
        expected_count=len(documents),
    )
    if zip_issues:
        raise ValueError(f"Phase 1 ZIP validation failed with {len(zip_issues)} issue(s)")
    return {
        "documents": len(documents),
        "predictions": len(predictions),
        "output_dir": str(config.output_dir),
        "zip_path": str(config.zip_path),
        "assertion_policy": config.assertion_policy,
        "candidate_policy": config.candidate_policy,
        "pipeline_config": str(config.pipeline_config_path)
        if config.pipeline_config_path is not None
        else None,
        "validation_issues": 0,
    }


def build_phase1_factory_config(config: Phase1BenchmarkConfig) -> PipelineFactoryConfig:
    """Compose a Phase 1 runner from a reusable pipeline profile.

    A benchmark should exercise the same terminology/search/NER composition as production.
    The profile is loaded here, at the benchmark boundary, so the core runner does not grow
    task-specific loading logic. CLI paths still override the profile's primary dictionary and
    abbreviation files; additional mined recognition sources and SQLite overlays are preserved.
    """

    if config.pipeline_config_path is None:
        factory_config = PipelineFactoryConfig(
            recognition_dictionary_path=str(config.dictionary_path),
            abbreviation_path=str(config.abbreviation_path),
        )
    else:
        payload = yaml.safe_load(
            config.pipeline_config_path.read_text(encoding="utf-8")
        ) or {}
        if not isinstance(payload, dict):
            raise ValueError("Phase 1 pipeline config must be a YAML mapping")
        factory_config = PipelineFactoryConfig.from_mapping(payload)

    # Phase 1 output has no relation field. Keep every other profile component intact while
    # avoiding needless relation extraction/validation work during submission generation.
    return replace(
        factory_config,
        recognition_dictionary_path=str(config.dictionary_path),
        abbreviation_path=str(config.abbreviation_path),
        options=replace(
            factory_config.options,
            enable_relations=False,
            enable_relation_kg_validation=False,
        ),
    )
