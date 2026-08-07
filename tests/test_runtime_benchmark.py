from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from clingrounder.evaluation.runtime_benchmark import (
    analyze_runtime_run,
    compare_runtime_runs,
    write_runtime_benchmark,
)


def test_runtime_benchmark_aggregates_counters_and_compares_outputs(tmp_path: Path) -> None:
    baseline_dir = _write_run(tmp_path / "baseline", docs_per_second=2.0)
    optimized_dir = _write_run(tmp_path / "optimized", docs_per_second=4.0)

    baseline = analyze_runtime_run(baseline_dir)
    optimized = analyze_runtime_run(optimized_dir)
    report = compare_runtime_runs([("baseline", baseline), ("optimized", optimized)])

    assert baseline["trace_count"] == 2
    assert baseline["stages"][0]["name"] == "candidate_generation"
    assert baseline["stages"][0]["counters"]["generated_candidates"] == 7
    assert report["runs"][1]["throughput_ratio"] == 2.0
    assert report["runs"][1]["output_identical_to_baseline"] is True

    write_runtime_benchmark(report, tmp_path / "report")
    assert (tmp_path / "report/runtime_benchmark.json").exists()
    assert "optimized" in (tmp_path / "report/runtime_benchmark.md").read_text()


def _write_run(path: Path, *, docs_per_second: float) -> Path:
    artifact_dir = path / "phase1"
    artifact_dir.mkdir(parents=True)
    (path / "run_manifest.json").write_text(
        json.dumps({"run_id": path.name, "content_hash": path.name}), encoding="utf-8"
    )
    (artifact_dir / "full_runtime.json").write_text(
        json.dumps(
            {
                "documents_per_second": docs_per_second,
                "initialization_ms": 10.0,
                "processing_ms": 20.0,
                "total_ms": 30.0,
            }
        ),
        encoding="utf-8",
    )
    traces = [
        {
            "document_id": "1",
            "bottleneck_stage": "candidate_generation",
            "stages": [
                {
                    "name": "candidate_generation",
                    "elapsed_ms": 5.0,
                    "counters": {"generated_candidates": 3},
                }
            ],
        },
        {
            "document_id": "2",
            "bottleneck_stage": "candidate_generation",
            "stages": [
                {
                    "name": "candidate_generation",
                    "elapsed_ms": 7.0,
                    "counters": {"generated_candidates": 4},
                }
            ],
        },
    ]
    (artifact_dir / "full_traces.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in traces), encoding="utf-8"
    )
    with ZipFile(artifact_dir / "full_output.zip", "w", ZIP_DEFLATED) as archive:
        archive.writestr("output/1.json", "[]\n")
    return path
