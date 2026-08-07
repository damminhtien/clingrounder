"""Isolated startup and peak-RSS benchmark for terminology repositories."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.terminology.sqlite_repository import SQLiteTerminologyRepository

__all__ = ["benchmark_terminology_repositories"]


def benchmark_terminology_repositories(
    source_paths: tuple[str | Path, ...],
    index_path: str | Path,
) -> dict[str, Any]:
    """Run each backend in a fresh process so peak RSS remains comparable."""

    sources = tuple(str(Path(source)) for source in source_paths)
    if not sources:
        raise ValueError("At least one terminology source is required")
    index = str(Path(index_path))
    memory = _run_worker("memory", sources, index)
    sqlite = _run_worker("sqlite", sources, index)
    speedup = float(memory["startup_ms"]) / max(float(sqlite["startup_ms"]), 1e-9)
    rss_reduction = 1.0 - float(sqlite["peak_rss_mb"]) / max(
        float(memory["peak_rss_mb"]), 1e-9
    )
    return {
        "sources": list(sources),
        "index": index,
        "memory": memory,
        "sqlite": sqlite,
        "startup_speedup": speedup,
        "rss_reduction": rss_reduction,
        "acceptance": {
            "startup_at_least_2x": speedup >= 2.0,
            "rss_reduction_at_least_40_percent": rss_reduction >= 0.40,
        },
    }


def _run_worker(backend: str, sources: tuple[str, ...], index: str) -> dict[str, Any]:
    source_arguments = [argument for source in sources for argument in ("--source", source)]
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "clingrounder.terminology.benchmark",
            "--worker",
            backend,
            *source_arguments,
            "--index",
            index,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(process.stdout)
    if not isinstance(payload, dict):
        raise ValueError("Benchmark worker returned a non-object payload")
    return payload


def _measure_worker(
    backend: str,
    sources: tuple[str, ...],
    index: str,
) -> dict[str, float | str]:
    started = perf_counter()
    if backend == "memory":
        entries = [
            entry
            for source in sources
            for entry in DictionaryStore.load_entries_jsonl(source)
        ]
        memory_repository = DictionaryStore(entries)
        concept_count = len(memory_repository.entries)
        memory_repository.exact_lookup("metformin")
    else:
        sqlite_repository = SQLiteTerminologyRepository(
            index,
            expected_source_paths=sources,
        )
        concept_count = int(sqlite_repository.metadata["concept_count"])
        sqlite_repository.exact_lookup("metformin")
    startup_ms = (perf_counter() - started) * 1000.0
    return {
        "backend": backend,
        "startup_ms": startup_ms,
        "peak_rss_mb": _peak_rss_mb(),
        "concept_count": float(concept_count),
    }


def _peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--worker", choices=("memory", "sqlite"))
    parser.add_argument("--output")
    args = parser.parse_args()
    sources = tuple(args.source)

    if args.worker:
        payload: dict[str, Any] = _measure_worker(args.worker, sources, args.index)
    else:
        payload = benchmark_terminology_repositories(sources, args.index)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    _main()
