from __future__ import annotations

import os
import threading
import traceback
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.pipeline.runner import PipelineRunResult, PipelineRunner
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction


ParallelBackend = Literal["serial", "thread", "process"]


@dataclass(frozen=True)
class ParallelBatchOptions:
    backend: ParallelBackend = "process"
    max_workers: int | None = None
    chunksize: int = 4
    fail_fast: bool = True


@dataclass(frozen=True)
class DocumentProcessingError:
    index: int
    document_id: str
    error_type: str
    message: str
    traceback: str


class ParallelBatchError(RuntimeError):
    def __init__(self, errors: list[DocumentProcessingError]) -> None:
        self.errors = errors
        summary = ", ".join(f"{error.document_id}: {error.message}" for error in errors[:5])
        suffix = f" and {len(errors) - 5} more" if len(errors) > 5 else ""
        super().__init__(
            f"{len(errors)} document(s) failed during parallel batch: {summary}{suffix}"
        )


_WORKER_RUNNER: PipelineRunner | None = None
_THREAD_LOCAL = threading.local()


def run_batch_with_trace_parallel(
    documents: list[ClinicalDocument],
    *,
    dictionary_path: str | Path = "data/dictionaries/seed_concepts.jsonl",
    abbreviation_path: str | Path = "data/dictionaries/abbreviations.jsonl",
    pipeline_version: str = "0.1.0",
    pipeline_options: PipelineOptions | None = None,
    parallel_options: ParallelBatchOptions | None = None,
) -> list[PipelineRunResult]:
    global _WORKER_RUNNER
    options = parallel_options or ParallelBatchOptions()
    pipeline_options = pipeline_options or PipelineOptions()
    worker_count = _effective_worker_count(options, len(documents))
    if not documents:
        return []
    if options.backend == "serial" or worker_count <= 1:
        runner = PipelineRunner(
            dictionary_path=dictionary_path,
            abbreviation_path=abbreviation_path,
            pipeline_version=pipeline_version,
            options=pipeline_options,
        )
        return [runner.process_document_with_trace(document) for document in documents]

    indexed_documents = list(enumerate(documents))
    initializer_args = (
        str(dictionary_path),
        str(abbreviation_path),
        pipeline_version,
        pipeline_options,
    )
    worker_fn = _process_indexed_document if options.fail_fast else _process_indexed_document_safe
    results: list[PipelineRunResult | None] = [None] * len(documents)
    errors: list[DocumentProcessingError] = []

    executor_context: Executor
    if options.backend == "thread":
        _WORKER_RUNNER = PipelineRunner(
            dictionary_path=dictionary_path,
            abbreviation_path=abbreviation_path,
            pipeline_version=pipeline_version,
            options=pipeline_options,
        )
        executor_context = ThreadPoolExecutor(max_workers=worker_count)
    else:
        executor_context = ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=initializer_args,
        )

    with executor_context as executor:
        for index, payload in executor.map(
            worker_fn, indexed_documents, chunksize=max(1, options.chunksize)
        ):
            if isinstance(payload, DocumentProcessingError):
                errors.append(payload)
            else:
                results[index] = payload

    if errors:
        raise ParallelBatchError(errors)
    return [_require_result(index, result) for index, result in enumerate(results)]


def run_batch_parallel(
    documents: list[ClinicalDocument],
    *,
    dictionary_path: str | Path = "data/dictionaries/seed_concepts.jsonl",
    abbreviation_path: str | Path = "data/dictionaries/abbreviations.jsonl",
    pipeline_version: str = "0.1.0",
    pipeline_options: PipelineOptions | None = None,
    parallel_options: ParallelBatchOptions | None = None,
) -> list[ClinicalPrediction]:
    return [
        result.prediction
        for result in run_batch_with_trace_parallel(
            documents,
            dictionary_path=dictionary_path,
            abbreviation_path=abbreviation_path,
            pipeline_version=pipeline_version,
            pipeline_options=pipeline_options,
            parallel_options=parallel_options,
        )
    ]


def _init_worker(
    dictionary_path: str,
    abbreviation_path: str,
    pipeline_version: str,
    pipeline_options: PipelineOptions,
) -> None:
    global _WORKER_RUNNER
    runner = PipelineRunner(
        dictionary_path=dictionary_path,
        abbreviation_path=abbreviation_path,
        pipeline_version=pipeline_version,
        options=pipeline_options,
    )
    _WORKER_RUNNER = runner
    _THREAD_LOCAL.runner = runner


def _process_indexed_document(item: tuple[int, ClinicalDocument]) -> tuple[int, PipelineRunResult]:
    index, document = item
    return index, _runner().process_document_with_trace(document)


def _process_indexed_document_safe(
    item: tuple[int, ClinicalDocument],
) -> tuple[int, PipelineRunResult | DocumentProcessingError]:
    index, document = item
    try:
        return index, _runner().process_document_with_trace(document)
    except Exception as error:  # pragma: no cover - exercised by integration failures.
        return index, DocumentProcessingError(
            index=index,
            document_id=document.document_id,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
        )


def _runner() -> PipelineRunner:
    runner = getattr(_THREAD_LOCAL, "runner", None) or _WORKER_RUNNER
    if runner is None:
        raise RuntimeError("Parallel worker runner was not initialized.")
    return runner


def _effective_worker_count(options: ParallelBatchOptions, document_count: int) -> int:
    if document_count <= 0:
        return 0
    if options.max_workers is not None:
        if options.max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        return min(options.max_workers, document_count)
    return min(os.cpu_count() or 1, document_count)


def _require_result(index: int, result: PipelineRunResult | None) -> PipelineRunResult:
    if result is None:
        raise RuntimeError(f"Missing parallel result for document index {index}.")
    return result
