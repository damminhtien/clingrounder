from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline.parallel_batch import ParallelBatchOptions, run_batch_with_trace_parallel
from medical_kg_nlp.schema.document import ClinicalDocument


def test_parallel_batch_thread_backend_preserves_input_order_and_traces() -> None:
    documents = _two_sample_documents()

    results = run_batch_with_trace_parallel(
        documents,
        parallel_options=ParallelBatchOptions(backend="thread", max_workers=2, chunksize=1),
    )

    assert [result.prediction.document_id for result in results] == ["sample_001", "sample_002"]
    assert all(result.trace.document_id == result.prediction.document_id for result in results)
    assert all(result.trace.bottleneck() is not None for result in results)


def test_parallel_batch_process_backend_preserves_input_order() -> None:
    documents = _two_sample_documents()

    results = run_batch_with_trace_parallel(
        documents,
        parallel_options=ParallelBatchOptions(backend="process", max_workers=2, chunksize=1),
    )

    assert [result.prediction.document_id for result in results] == ["sample_001", "sample_002"]
    assert [result.trace.document_id for result in results] == ["sample_001", "sample_002"]


def test_run_pipeline_cli_accepts_parallel_workers(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_pipeline.py",
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
