"""Instance-scoped batch execution for serial, thread, and process backends."""

from __future__ import annotations

import os
import pickle
import threading
import traceback
from collections.abc import Callable, MutableMapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter
from typing import Literal, TypeAlias

from medical_kg_nlp.pipeline.factory import PipelineFactory, PipelineFactoryConfig
from medical_kg_nlp.pipeline.runner import PipelineRunResult, PipelineRunner
from medical_kg_nlp.pipeline.runtime import RuntimeCapabilities
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction

__all__ = [
    "ParallelBackend",
    "ParallelBatchError",
    "ParallelBatchOptions",
    "PipelineBatchExecutor",
    "run_batch_parallel",
    "run_batch_with_trace_parallel",
]

ParallelBackend = Literal["serial", "thread", "process"]
RunnerFactory: TypeAlias = Callable[[], PipelineRunner]


@dataclass(frozen=True)
class ParallelBatchOptions:
    backend: ParallelBackend = "process"
    max_workers: int | None = None
    chunksize: int = 4
    fail_fast: bool = True
    runtime_capabilities: RuntimeCapabilities = field(default_factory=RuntimeCapabilities)

    def __post_init__(self) -> None:
        if self.chunksize < 1:
            raise ValueError("chunksize must be at least 1.")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be at least 1.")


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


# PROCESS-LOCAL ONLY: this state is initialized separately in each worker process. Thread
# execution never reads or writes it; keeping the boundary explicit avoids cross-batch leakage.
_PROCESS_RUNNER: PipelineRunner | None = None


class PipelineBatchExecutor:
    """Own one batch runtime and all resources used to execute it.

    Thread workers either share the executor's runner when explicitly declared thread-safe or
    lazily create one runner per worker thread. Process workers create one runner in their own
    process through the initializer, so model objects are never serialized per document.
    """

    def __init__(
        self,
        runner_factory: RunnerFactory,
        options: ParallelBatchOptions,
    ) -> None:
        self._runner_factory = runner_factory
        self.options = options
        self.capabilities = options.runtime_capabilities
        self._closed = False
        self._thread_local = threading.local()
        self._runner_lock = threading.Lock()
        self._thread_runners: list[PipelineRunner] = []
        self._shared_thread_runner: PipelineRunner | None = None
        self._serial_runner: PipelineRunner | None = None
        self._executor: Executor | None = None

        if options.backend not in {"serial", "thread", "process"}:
            raise ValueError(f"Unsupported parallel backend {options.backend!r}")
        if options.backend == "thread" and self.capabilities.thread_safe:
            self._shared_thread_runner = self._runner_factory()
        elif options.backend == "serial":
            self._serial_runner = self._runner_factory()
        elif options.backend == "process":
            if not self.capabilities.process_safe:
                raise ValueError("Process backend requires process_safe runtime capabilities.")
            if (
                self.capabilities.device_kind in {"cuda", "mps"}
                and _configured_worker_count(options) > 1
            ):
                raise ValueError(
                    "Process parallelism is disabled for GPU-backed pipelines; use serial or "
                    "a declared thread-safe shared runtime."
                )
            try:
                pickle.dumps(runner_factory)
            except (pickle.PicklingError, TypeError, AttributeError) as error:
                raise ValueError(
                    "Process backend requires a picklable runner_factory."
                ) from error

        if options.backend == "thread":
            self._executor = ThreadPoolExecutor(max_workers=options.max_workers)
        elif options.backend == "process":
            self._executor = ProcessPoolExecutor(
                max_workers=options.max_workers,
                initializer=_init_process_worker,
                initargs=(self._runner_factory,),
            )

    def run(self, documents: Sequence[ClinicalDocument]) -> list[PipelineRunResult]:
        """Process documents in input order while preserving configured error semantics."""

        self._ensure_open()
        indexed_documents = list(enumerate(documents))
        if not indexed_documents:
            return []

        if self.options.backend == "serial":
            runner = self._serial_runner
            if runner is None:  # pragma: no cover - constructor invariant.
                raise RuntimeError("Serial runner was not initialized.")
            return [runner.process_document_with_trace(document) for document in documents]

        executor = self._executor
        if executor is None:  # pragma: no cover - constructor invariant.
            raise RuntimeError("Parallel executor was not initialized.")
        worker = self._thread_worker if self.options.backend == "thread" else _process_worker
        safe_worker = (
            self._thread_worker_safe
            if self.options.backend == "thread"
            else _process_worker_safe
        )
        results: list[PipelineRunResult | None] = [None] * len(indexed_documents)
        errors: list[DocumentProcessingError] = []
        worker_fn = worker if self.options.fail_fast else safe_worker

        for index, payload in executor.map(
            worker_fn,
            indexed_documents,
            chunksize=max(1, self.options.chunksize),
        ):
            if isinstance(payload, DocumentProcessingError):
                errors.append(payload)
            else:
                results[index] = payload
        if errors:
            raise ParallelBatchError(errors)
        return [_require_result(index, result) for index, result in enumerate(results)]

    def close(self) -> None:
        """Release workers and close every runner owned by this executor."""

        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        runners: list[PipelineRunner] = []
        if self._serial_runner is not None:
            runners.append(self._serial_runner)
        if self._shared_thread_runner is not None:
            runners.append(self._shared_thread_runner)
        with self._runner_lock:
            runners.extend(self._thread_runners)
            self._thread_runners.clear()
        for runner in runners:
            close = getattr(runner, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "PipelineBatchExecutor":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback_value: object) -> None:
        self.close()

    def _thread_worker(
        self,
        item: tuple[int, ClinicalDocument],
    ) -> tuple[int, PipelineRunResult]:
        index, document = item
        return index, self._thread_runner().process_document_with_trace(document)

    def _thread_worker_safe(
        self,
        item: tuple[int, ClinicalDocument],
    ) -> tuple[int, PipelineRunResult | DocumentProcessingError]:
        index, document = item
        try:
            return self._thread_worker(item)
        except Exception as error:  # pragma: no cover - exercised by integration failures.
            return index, _processing_error(index, document, error)

    def _thread_runner(self) -> PipelineRunner:
        if self._shared_thread_runner is not None:
            return self._shared_thread_runner
        runner = getattr(self._thread_local, "runner", None)
        if runner is None:
            runner = self._runner_factory()
            self._thread_local.runner = runner
            with self._runner_lock:
                self._thread_runners.append(runner)
        return runner

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PipelineBatchExecutor is closed.")


def run_batch_with_trace_parallel(
    documents: Sequence[ClinicalDocument],
    *,
    factory_config: PipelineFactoryConfig | None = None,
    parallel_options: ParallelBatchOptions | None = None,
    runtime_metrics: MutableMapping[str, object] | None = None,
) -> list[PipelineRunResult]:
    """Compatibility wrapper that owns and closes one executor for this call."""

    total_started = perf_counter()
    options = parallel_options or ParallelBatchOptions()
    resolved_factory_config = factory_config or PipelineFactoryConfig()
    runner_factory = partial(PipelineFactory.from_config, resolved_factory_config)
    if not documents:
        _record_runtime_metrics(
            runtime_metrics,
            backend=options.backend,
            worker_count=0,
            document_count=0,
            initialization_ms=0.0,
            processing_ms=0.0,
            total_started=total_started,
            worker_initialization_in_processing=False,
        )
        return []
    worker_count = _effective_worker_count(options, len(documents))
    if options.backend == "serial" or worker_count <= 1:
        serial_options = ParallelBatchOptions(
            backend="serial",
            max_workers=1,
            chunksize=options.chunksize,
            fail_fast=options.fail_fast,
            runtime_capabilities=options.runtime_capabilities,
        )
        initialization_started = perf_counter()
        with PipelineBatchExecutor(runner_factory, serial_options) as executor:
            initialization_ms = (perf_counter() - initialization_started) * 1000.0
            processing_started = perf_counter()
            results = executor.run(documents)
            processing_ms = (perf_counter() - processing_started) * 1000.0
        _record_runtime_metrics(
            runtime_metrics,
            backend="serial",
            worker_count=1,
            document_count=len(documents),
            initialization_ms=initialization_ms,
            processing_ms=processing_ms,
            total_started=total_started,
            worker_initialization_in_processing=False,
        )
        return results

    initialization_started = perf_counter()
    with PipelineBatchExecutor(runner_factory, options) as executor:
        parallel_initialization_ms: float | None = (
            (perf_counter() - initialization_started) * 1000.0
            if options.backend == "thread"
            else None
        )
        processing_started = perf_counter()
        results = executor.run(documents)
        processing_ms = (perf_counter() - processing_started) * 1000.0
    _record_runtime_metrics(
        runtime_metrics,
        backend=options.backend,
        worker_count=worker_count,
        document_count=len(documents),
        initialization_ms=parallel_initialization_ms,
        processing_ms=processing_ms,
        total_started=total_started,
        worker_initialization_in_processing=options.backend == "process",
    )
    return results


def run_batch_parallel(
    documents: Sequence[ClinicalDocument],
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


def _init_process_worker(runner_factory: RunnerFactory) -> None:
    global _PROCESS_RUNNER
    _PROCESS_RUNNER = runner_factory()


def _process_worker(item: tuple[int, ClinicalDocument]) -> tuple[int, PipelineRunResult]:
    index, document = item
    runner = _PROCESS_RUNNER
    if runner is None:
        raise RuntimeError("Process worker runner was not initialized.")
    return index, runner.process_document_with_trace(document)


def _process_worker_safe(
    item: tuple[int, ClinicalDocument],
) -> tuple[int, PipelineRunResult | DocumentProcessingError]:
    index, document = item
    try:
        return _process_worker(item)
    except Exception as error:  # pragma: no cover - exercised by integration failures.
        return index, _processing_error(index, document, error)


def _processing_error(
    index: int,
    document: ClinicalDocument,
    error: Exception,
) -> DocumentProcessingError:
    return DocumentProcessingError(
        index=index,
        document_id=document.document_id,
        error_type=type(error).__name__,
        message=str(error),
        traceback=traceback.format_exc(),
    )


def _effective_worker_count(options: ParallelBatchOptions, document_count: int) -> int:
    if document_count <= 0:
        return 0
    if options.max_workers is not None:
        if options.max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        return min(options.max_workers, document_count)
    return min(os.cpu_count() or 1, document_count)


def _configured_worker_count(options: ParallelBatchOptions) -> int:
    return options.max_workers if options.max_workers is not None else os.cpu_count() or 1


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
