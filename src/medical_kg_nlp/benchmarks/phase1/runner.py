"""Reproducible Phase 1 benchmark build service used by the consolidated CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
from medical_kg_nlp.pipeline.options import PipelineOptions

__all__ = ["BenchmarkExportPolicy", "Phase1BenchmarkConfig", "run_phase1_benchmark"]

BenchmarkExportPolicy = Literal["empty", "pipeline"]


@dataclass(frozen=True)
class Phase1BenchmarkConfig:
    """Inputs and conservative export policies for one benchmark artifact."""

    input_dir: Path
    output_dir: Path
    zip_path: Path
    dictionary_path: Path
    abbreviation_path: Path
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
        factory_config=PipelineFactoryConfig(
            recognition_dictionary_path=str(config.dictionary_path),
            abbreviation_path=str(config.abbreviation_path),
            # Phase 1 has no relation output. Avoid paying for relation extraction and
            # validation while preserving all entity, assertion, and linking stages.
            options=PipelineOptions(
                enable_relations=False,
                enable_relation_kg_validation=False,
            ),
        ),
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
        "validation_issues": 0,
    }
