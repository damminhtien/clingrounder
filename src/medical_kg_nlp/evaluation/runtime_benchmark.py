from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


def analyze_runtime_run(run_dir: str | Path) -> dict[str, Any]:
    """Summarize one instrumented pipeline run without loading prediction payloads."""
    root = Path(run_dir)
    runtime_path = _find_single(root, "*runtime.json")
    traces_path = _find_single(root, "*traces.jsonl")
    manifest_path = root / "run_manifest.json"
    zip_paths = sorted(root.rglob("*output.zip"))

    runtime = _read_json(runtime_path)
    stage_elapsed: dict[str, list[float]] = defaultdict(list)
    stage_counters: dict[str, Counter[str]] = defaultdict(Counter)
    bottlenecks: Counter[str] = Counter()
    trace_count = 0

    with traces_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            trace_count += 1
            bottleneck = trace.get("bottleneck_stage")
            if bottleneck:
                bottlenecks[str(bottleneck)] += 1
            for stage in trace.get("stages", []):
                name = str(stage["name"])
                stage_elapsed[name].append(float(stage.get("elapsed_ms", 0.0)))
                stage_counters[name].update(
                    {str(key): int(value) for key, value in stage.get("counters", {}).items()}
                )

    # Aggregate trace counters by summation. Merging dictionaries would silently retain only the
    # final document and becomes misleading as batch size grows.
    stages = []
    for name, elapsed in stage_elapsed.items():
        total_ms = sum(elapsed)
        stages.append(
            {
                "name": name,
                "calls": len(elapsed),
                "total_ms": round(total_ms, 6),
                "avg_ms": round(total_ms / len(elapsed), 6),
                "max_ms": round(max(elapsed), 6),
                "counters": dict(sorted(stage_counters[name].items())),
            }
        )
    stages.sort(key=lambda row: cast(float, row["total_ms"]), reverse=True)

    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    zip_path = zip_paths[0] if len(zip_paths) == 1 else None
    return {
        "run_dir": str(root),
        "run_id": manifest.get("run_id", root.name),
        "content_hash": manifest.get("content_hash"),
        "runtime": runtime,
        "trace_count": trace_count,
        "stages": stages,
        "bottleneck_document_counts": dict(bottlenecks.most_common()),
        "output_zip": str(zip_path) if zip_path else None,
        "output_sha256": _sha256(zip_path) if zip_path else None,
    }


def compare_runtime_runs(
    named_runs: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compare runs against the first entry, which acts as the immutable baseline."""
    if not named_runs:
        raise ValueError("At least one runtime run is required")
    baseline_name, baseline = named_runs[0]
    baseline_runtime = _mapping(baseline["runtime"])
    baseline_throughput = float(baseline_runtime["documents_per_second"])
    baseline_sha = baseline.get("output_sha256")
    comparisons = []
    for name, run in named_runs:
        runtime = _mapping(run["runtime"])
        throughput = float(runtime["documents_per_second"])
        comparisons.append(
            {
                "name": name,
                "run_id": run.get("run_id"),
                "documents_per_second": throughput,
                "throughput_ratio": round(throughput / baseline_throughput, 6),
                "initialization_ms": float(runtime["initialization_ms"]),
                "processing_ms": float(runtime["processing_ms"]),
                "total_ms": float(runtime["total_ms"]),
                "output_sha256": run.get("output_sha256"),
                "output_identical_to_baseline": bool(
                    baseline_sha and run.get("output_sha256") == baseline_sha
                ),
                "bottleneck_stage": _top_stage(run),
            }
        )
    return {
        "baseline": baseline_name,
        "runs": comparisons,
        "details": {name: dict(run) for name, run in named_runs},
    }


def write_runtime_benchmark(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Runtime Benchmark",
        "",
        f"Baseline: `{report['baseline']}`",
        "",
        "| Run | Docs/s | Ratio | Init ms | Process ms | Bottleneck | Same output |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["runs"]:
        lines.append(
            "| {name} | {documents_per_second:.3f} | {throughput_ratio:.3f} | "
            "{initialization_ms:.1f} | {processing_ms:.1f} | {bottleneck_stage} | "
            "{output_identical_to_baseline} |".format(**row)
        )
    (output / "runtime_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_single(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern!r} under {root}, found {len(matches)}")
    return matches[0]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping, got {type(value).__name__}")
    return value


def _top_stage(run: Mapping[str, Any]) -> str | None:
    stages = run.get("stages", [])
    if not isinstance(stages, list) or not stages:
        return None
    return str(stages[0]["name"])
