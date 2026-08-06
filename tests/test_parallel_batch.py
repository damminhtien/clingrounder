from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline import (
    PipelineBatchExecutor,
    PipelineComponents,
    PipelineFactory,
    PipelineConfig,
    PipelineOptions,
    PipelineRunner,
    RuntimeCapabilities,
)
from medical_kg_nlp.pipeline.parallel_batch import (
    ParallelBatchError,
    ParallelBatchOptions,
)
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument


def test_parallel_batch_thread_backend_preserves_input_order_and_traces() -> None:
    documents = _two_sample_documents()
    executor = PipelineBatchExecutor(
        partial(PipelineFactory.from_config, PipelineConfig()),
        ParallelBatchOptions(backend="thread", max_workers=2, chunksize=1),
    )
    try:
        results = executor.run(documents)
    finally:
        executor.close()

    assert [result.prediction.document_id for result in results] == ["sample_001", "sample_002"]
    assert all(result.trace.document_id == result.prediction.document_id for result in results)
    assert all(result.trace.bottleneck() is not None for result in results)


@pytest.mark.integration
def test_parallel_batch_process_backend_preserves_input_order() -> None:
    documents = _two_sample_documents()

    with PipelineBatchExecutor(
        partial(PipelineFactory.from_config, PipelineConfig()),
        ParallelBatchOptions(backend="process", max_workers=2, chunksize=1),
    ) as executor:
        results = executor.run(documents)

    assert [result.prediction.document_id for result in results] == ["sample_001", "sample_002"]
    assert [result.trace.document_id for result in results] == ["sample_001", "sample_002"]


@pytest.mark.integration
def test_run_pipeline_cli_accepts_parallel_workers(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.jsonl"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "medical_kg_nlp.cli",
            "pipeline",
            "run",
            "--config",
            "configs/pipeline/clinical-baseline.yaml",
            "--input",
            "data/samples/sample_notes.jsonl",
            "--output",
            str(output_path),
            "--workers",
            "2",
            "--parallel-backend",
            "thread",
            "--chunksize",
            "1",
        ],
        check=True,
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["document_id"] for row in rows] == ["sample_001"]


def test_executors_with_different_versions_do_not_leak_configuration() -> None:
    documents = [ClinicalDocument(document_id=f"doc-{index}", text="text") for index in range(4)]
    executor_a = _new_executor("version-a")
    executor_b = _new_executor("version-b")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(executor.run, documents)
                for executor in (executor_a, executor_b)
            ]
            results_a, results_b = [future.result() for future in futures]
        assert {result.prediction.metadata.pipeline_version for result in results_a} == {
            "version-a"
        }
        assert {result.prediction.metadata.pipeline_version for result in results_b} == {
            "version-b"
        }
    finally:
        executor_a.close()
        executor_b.close()


def test_thread_executor_preserves_order_and_owns_unsafe_thread_runners() -> None:
    documents = [ClinicalDocument(document_id=f"doc-{index}", text="text") for index in range(8)]
    close_counter: list[str] = []
    executor = PipelineBatchExecutor(
        partial(_build_runner, "thread-safe-test", close_counter),
        ParallelBatchOptions(
            backend="thread",
            max_workers=2,
            runtime_capabilities=RuntimeCapabilities(thread_safe=False),
        ),
    )
    try:
        results = executor.run(documents)
        assert [result.prediction.document_id for result in results] == [
            document.document_id for document in documents
        ]
    finally:
        executor.close()
    assert len(close_counter) >= 1


def test_thread_safe_executor_creates_one_shared_runner() -> None:
    close_counter: list[str] = []
    executor = PipelineBatchExecutor(
        partial(_build_runner, "shared", close_counter),
        ParallelBatchOptions(
            backend="thread",
            max_workers=3,
            runtime_capabilities=RuntimeCapabilities(thread_safe=True),
        ),
    )
    try:
        executor.run(
            [ClinicalDocument(document_id=f"doc-{index}", text="text") for index in range(6)]
        )
    finally:
        executor.close()
    assert close_counter == ["shared"]


def test_executor_close_is_idempotent_and_blocks_future_runs() -> None:
    executor = _new_executor("close-test")
    executor.close()
    executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        executor.run([ClinicalDocument(document_id="doc", text="text")])


def test_error_collection_preserves_successful_processing_diagnostics() -> None:
    executor = PipelineBatchExecutor(
        partial(_build_runner, "failure-test", [], True),
        ParallelBatchOptions(backend="thread", max_workers=2, fail_fast=False),
    )
    try:
        # The extractor raises only for one document; the other document still executes.
        with pytest.raises(ParallelBatchError) as error_info:
            executor.run(
                [
                    ClinicalDocument(document_id="ok", text="text"),
                    ClinicalDocument(document_id="bad", text="FAIL"),
                ]
            )
        assert [error.document_id for error in error_info.value.errors] == ["bad"]
    finally:
        executor.close()


def test_fail_fast_propagates_the_worker_exception() -> None:
    executor = PipelineBatchExecutor(
        partial(_build_runner, "fail-fast", [], True),
        ParallelBatchOptions(backend="thread", max_workers=2, fail_fast=True),
    )
    try:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            executor.run([ClinicalDocument(document_id="bad", text="FAIL")])
    finally:
        executor.close()


def test_process_backend_rejects_unsafe_gpu_parallelism() -> None:
    with pytest.raises(ValueError, match="GPU-backed"):
        PipelineBatchExecutor(
            partial(_build_runner, "gpu", []),
            ParallelBatchOptions(
                backend="process",
                max_workers=2,
                runtime_capabilities=RuntimeCapabilities(device_kind="cuda"),
            ),
        )


@pytest.mark.integration
def test_process_executor_initializes_one_runner_per_worker() -> None:
    executor = PipelineBatchExecutor(
        partial(_build_runner, "process-test", []),
        ParallelBatchOptions(backend="process", max_workers=2, chunksize=1),
    )
    try:
        documents = [
            ClinicalDocument(document_id=f"doc-{index}", text="text") for index in range(4)
        ]
        results = executor.run(documents)
        assert [result.prediction.document_id for result in results] == [
            document.document_id for document in documents
        ]
        assert {result.prediction.metadata.pipeline_version for result in results} == {
            "process-test"
        }
    finally:
        executor.close()


def _new_executor(version: str) -> PipelineBatchExecutor:
    return PipelineBatchExecutor(
        partial(_build_runner, version, []),
        ParallelBatchOptions(
            backend="thread",
            max_workers=2,
            runtime_capabilities=RuntimeCapabilities(thread_safe=False),
        ),
    )


def _build_runner(
    version: str,
    close_counter: list[str],
    fail_on_marker: bool = False,
) -> PipelineRunner:
    return _ClosablePipelineRunner(
        PipelineComponents(
            entity_extractor=_EmptyExtractor(fail_on_marker=fail_on_marker),
            options=_minimal_options(),
            pipeline_version=version,
            runtime_capabilities=RuntimeCapabilities(
                thread_safe=False,
                process_safe=True,
            ),
        ),
        close_counter,
    )


def _minimal_options() -> PipelineOptions:
    return PipelineOptions(
        enable_context=False,
        enable_linking=False,
        enable_candidate_reranking=False,
        enable_entity_kg_validation=False,
        enable_relations=False,
        enable_relation_kg_validation=False,
    )


@dataclass(frozen=True)
class _EmptyExtractor:
    fail_on_marker: bool = False

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        if self.fail_on_marker and source_text == "FAIL":
            raise RuntimeError("synthetic failure")
        return []


class _ClosablePipelineRunner(PipelineRunner):
    def __init__(self, components: PipelineComponents, close_counter: list[str]) -> None:
        super().__init__(components)
        self._close_counter = close_counter

    def close(self) -> None:
        self._close_counter.append(self.components.pipeline_version)


def _two_sample_documents() -> list[ClinicalDocument]:
    document = SyntheticDatasetAdapter().load_documents("data/samples/sample_notes.jsonl")[0]
    return [
        document,
        ClinicalDocument(
            document_id="sample_002",
            text=document.text,
            metadata=dict(document.metadata),
        ),
    ]
