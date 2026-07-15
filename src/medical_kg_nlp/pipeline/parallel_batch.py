from __future__ import annotations

import os
import threading
import traceback
from collections.abc import MutableMapping
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from medical_kg_nlp.pipeline.factory import PipelineFactory, PipelineFactoryConfig
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
    factory_config: PipelineFactoryConfig | None = None,
    parallel_options: ParallelBatchOptions | None = None,
    runtime_metrics: MutableMapping[str, object] | None = None,
) -> list[PipelineRunResult]:
    global _WORKER_RUNNER
    total_started = perf_counter()
    options = parallel_options or ParallelBatchOptions()
    resolved_factory_config = factory_config or PipelineFactoryConfig()
    worker_count = _effective_worker_count(options, len(documents))
    if not documents:
        _record_runtime_metrics(
            runtime_metrics,
            backend=options.backend,
            worker_count=worker_count,
            document_count=0,
            initialization_ms=0.0,
            processing_ms=0.0,
            total_started=total_started,
            worker_initialization_in_processing=False,
        )
        return []
    if options.backend == "serial" or worker_count <= 1:
        initialization_started = perf_counter()
        runner = PipelineFactory.from_config(resolved_factory_config)
        serial_initialization_ms = (perf_counter() - initialization_started) * 1000.0
        processing_started = perf_counter()
        serial_results = [
            runner.process_document_with_trace(document) for document in documents
        ]
        serial_processing_ms = (perf_counter() - processing_started) * 1000.0
        _record_runtime_metrics(
            runtime_metrics,
            backend="serial",
            worker_count=1,
            document_count=len(documents),
            initialization_ms=serial_initialization_ms,
            processing_ms=serial_processing_ms,
            total_started=total_started,
            worker_initialization_in_processing=False,
        )
        return serial_results

    indexed_documents = list(enumerate(documents))
    initializer_args = (resolved_factory_config,)
    worker_fn = _process_indexed_document if options.fail_fast else _process_indexed_document_safe
    results: list[PipelineRunResult | None] = [None] * len(documents)
    errors: list[DocumentProcessingError] = []

    executor_context: Executor
    initialization_ms: float | None
    worker_initialization_in_processing = False
    if options.backend == "thread":
        initialization_started = perf_counter()
        _WORKER_RUNNER = PipelineFactory.from_config(resolved_factory_config)
        initialization_ms = (perf_counter() - initialization_started) * 1000.0
        executor_context = ThreadPoolExecutor(max_workers=worker_count)
    else:
        initialization_ms = None
        worker_initialization_in_processing = True
        executor_context = ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=initializer_args,
        )

    processing_started = perf_counter()
    with executor_context as executor:
        for index, payload in executor.map(
            worker_fn, indexed_documents, chunksize=max(1, options.chunksize)
        ):
            if isinstance(payload, DocumentProcessingError):
                errors.append(payload)
            else:
                results[index] = payload
    processing_ms = (perf_counter() - processing_started) * 1000.0

    if errors:
        raise ParallelBatchError(errors)
    output = [_require_result(index, result) for index, result in enumerate(results)]
    _record_runtime_metrics(
        runtime_metrics,
        backend=options.backend,
        worker_count=worker_count,
        document_count=len(documents),
        initialization_ms=initialization_ms,
        processing_ms=processing_ms,
        total_started=total_started,
        worker_initialization_in_processing=worker_initialization_in_processing,
    )
    return output


def run_batch_parallel(
    documents: list[ClinicalDocument],
    *,
    factory_config: PipelineFactoryConfig | None = None,
    parallel_options: ParallelBatchOptions | None = None,
) -> list[ClinicalPrediction]:
    return [
        result.prediction
        for result in run_batch_with_trace_parallel(
            documents,
            factory_config=factory_config,
            parallel_options=parallel_options,
        )
    ]


def _init_worker(
    factory_config: PipelineFactoryConfig,
) -> None:
    global _WORKER_RUNNER
    # SCALING: each process owns its concrete components; thread-safe repositories may still
    # allocate thread-local read connections behind their port.
    runner = PipelineFactory.from_config(factory_config)
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


def _record_runtime_metrics(
    target: MutableMapping[str, object] | None,
    *,
    backend: ParallelBackend,
    worker_count: int,
    document_count: int,
    initialization_ms: float | None,
    processing_ms: float,
    total_started: float,
    worker_initialization_in_processing: bool,
) -> None:
    if target is None:
        return
    total_ms = (perf_counter() - total_started) * 1000.0
    target.update(
        {
            "backend": backend,
            "worker_count": worker_count,
            "document_count": document_count,
            "initialization_ms": initialization_ms,
            "processing_ms": processing_ms,
            "total_ms": total_ms,
            "documents_per_second": (
                document_count / (total_ms / 1000.0) if total_ms > 0.0 else 0.0
            ),
            "worker_initialization_in_processing": worker_initialization_in_processing,
        }
    )
