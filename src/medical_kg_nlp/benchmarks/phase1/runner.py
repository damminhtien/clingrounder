"""Reproducible Phase 1 benchmark build service used by the consolidated CLI."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

from medical_kg_nlp.benchmarks.phase1.phase1 import (
    load_phase1_text_documents,
    validate_phase1_submission_documents,
    validate_phase1_submission_zip,
    write_phase1_output_dir,
    zip_phase1_output_dir,
)
from medical_kg_nlp.benchmarks.phase1.assertion_overlays import (
    load_phase1_assertion_overlays,
)
from medical_kg_nlp.benchmarks.phase1.round2 import load_phase1_round2_documents
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.mining.io import load_documents
from medical_kg_nlp.pipeline.config_loader import ResolvedPipelineConfig
from medical_kg_nlp.pipeline.factory import PipelineFactory, PipelineConfig, TerminologyConfig
from medical_kg_nlp.pipeline.parallel_batch import (
    ParallelBatchOptions,
    ParallelBackend,
    PipelineBatchExecutor,
)
from medical_kg_nlp.schema.document import ClinicalDocument

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

    input_dir: Path | None
    output_dir: Path
    zip_path: Path
    dictionary_path: Path
    abbreviation_path: Path
    documents_path: Path | None = None
    expected_source_archive_sha256: str | None = None
    pipeline_config_path: Path | None = None
    validation_dictionary_paths: tuple[Path, ...] = ()
    assertion_overlay_path: Path | None = None
    assertion_policy: BenchmarkExportPolicy = "pipeline"
    candidate_policy: BenchmarkExportPolicy = "pipeline"
    max_candidates: int = 5
    backend: ParallelBackend = "process"
    workers: int = 1
    chunksize: int = 4


def run_phase1_benchmark(config: Phase1BenchmarkConfig) -> dict[str, Any]:
    """Run, release-validate, and deterministically archive a Phase 1 submission."""

    documents = _load_benchmark_documents(config)
    runner_factory = partial(
        PipelineFactory.from_config,
        build_phase1_factory_config(config),
    )
    with PipelineBatchExecutor(
        runner_factory,
        ParallelBatchOptions(
            backend=config.backend,
            max_workers=config.workers,
            chunksize=config.chunksize,
        ),
    ) as executor:
        results = executor.run(documents)
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
        assertion_overlays=load_phase1_assertion_overlays(config.assertion_overlay_path),
    )
    dictionary = _load_validation_dictionary(config)
    directory_issues = validate_phase1_submission_documents(
        documents,
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
        documents=documents,
        dictionary=dictionary,
        expected_count=len(documents),
    )
    if zip_issues:
        raise ValueError(f"Phase 1 ZIP validation failed with {len(zip_issues)} issue(s)")
    return {
        "documents": len(documents),
        "source": (
            {"kind": "mined_manifest", "path": str(config.documents_path)}
            if config.documents_path is not None
            else {"kind": "text_directory", "path": str(config.input_dir)}
        ),
        "predictions": len(predictions),
        "output_dir": str(config.output_dir),
        "zip_path": str(config.zip_path),
        "assertion_policy": config.assertion_policy,
        "candidate_policy": config.candidate_policy,
        "assertion_overlay_path": (
            str(config.assertion_overlay_path)
            if config.assertion_overlay_path is not None
            else None
        ),
        "pipeline_config": str(config.pipeline_config_path)
        if config.pipeline_config_path is not None
        else None,
        "validation_dictionaries": [str(path) for path in _validation_paths(config)],
        "validation_issues": 0,
    }


def _load_benchmark_documents(config: Phase1BenchmarkConfig) -> list[ClinicalDocument]:
    """Load one mutually exclusive source while retaining Round 2 privacy checks."""

    if (config.input_dir is None) == (config.documents_path is None):
        raise ValueError("Provide exactly one of input_dir or documents_path")
    if config.documents_path is None:
        assert config.input_dir is not None
        if config.expected_source_archive_sha256 is not None:
            raise ValueError("Source archive SHA-256 is valid only with documents_path")
        return load_phase1_text_documents(config.input_dir)
    if config.expected_source_archive_sha256 is None:
        raise ValueError("Round 2 documents_path requires expected_source_archive_sha256")
    return load_phase1_round2_documents(
        load_documents(config.documents_path),
        expected_archive_sha256=config.expected_source_archive_sha256,
    )


def build_phase1_factory_config(config: Phase1BenchmarkConfig) -> PipelineConfig:
    """Compose a Phase 1 runner from a reusable pipeline profile.

    A benchmark should exercise the same terminology/search/NER composition as production.
    The profile is loaded here, at the benchmark boundary, so the core runner does not grow
    task-specific loading logic. CLI paths still override the profile's primary dictionary and
    abbreviation files; additional mined recognition sources and SQLite overlays are preserved.
    """

    if config.pipeline_config_path is None:
        factory_config = PipelineConfig(
            terminology=TerminologyConfig(
                recognition_dictionary_path=str(config.dictionary_path),
                abbreviation_path=str(config.abbreviation_path),
            )
        )
    else:
        factory_config = ResolvedPipelineConfig.load(
            config.pipeline_config_path
        ).factory_config

    # Phase 1 output has no relation field. Keep every other profile component intact while
    # avoiding needless relation extraction/validation work during submission generation.
    return replace(
        factory_config,
        terminology=replace(
            factory_config.terminology,
            recognition_dictionary_path=str(config.dictionary_path),
            abbreviation_path=str(config.abbreviation_path),
        ),
        options=replace(
            factory_config.options,
            enable_relations=False,
            enable_relation_kg_validation=False,
        ),
    )


def _validation_paths(config: Phase1BenchmarkConfig) -> tuple[Path, ...]:
    """Return the code sources accepted by the release validator."""

    # The recognition dictionary may contain legacy competition codes that are not part of the
    # current TT06 release. Keep it in the validation view, then add any explicitly pinned full
    # sources for codes emitted by normalization.
    paths = (config.dictionary_path, *config.validation_dictionary_paths)
    return tuple(dict.fromkeys(paths))


def _load_validation_dictionary(config: Phase1BenchmarkConfig) -> DictionaryStore:
    """Build one validation view from one or more pinned canonical JSONL sources.

    Recognition and normalization deliberately have different storage footprints. A full profile
    may emit a code from TT06/RxNorm even though its recognition dictionary is intentionally tiny;
    release validation must therefore receive the same canonical code sources explicitly.
    """

    entries = []
    for path in _validation_paths(config):
        entries.extend(DictionaryStore.load_entries_jsonl(path))
    # Validation only needs the set of legal (system, code) pairs. Do not merge metadata here:
    # releases can intentionally reuse a concept ID while carrying different parent metadata
    # (for example a legacy seed row versus the current TT06 row).
    return DictionaryStore(entries)
