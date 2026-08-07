"""Task-neutral benchmark plugin commands."""

from __future__ import annotations

import argparse
import json

from clingrounder.benchmarks.registry import benchmark_plugins

from clingrounder.evaluation.promotion_benchmark import (
    compare_promotion_benchmarks,
    run_promotion_benchmark,
)

__all__ = [
    "compare_runtime_benchmark",
    "list_benchmarks",
    "run_runtime_benchmark",
]


def list_benchmarks(args: argparse.Namespace) -> int:
    """List installed benchmark plugins without importing benchmark implementations."""

    del args
    payload = [
        {"name": plugin.name, "summary": plugin.summary}
        for plugin in benchmark_plugins()
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_runtime_benchmark(args: argparse.Namespace) -> int:
    """Run the reusable runtime benchmark and persist its CI artifact."""

    report = run_promotion_benchmark(
        args.input,
        args.config,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def compare_runtime_benchmark(args: argparse.Namespace) -> int:
    """Compare two benchmark artifacts and fail when promotion gates fail."""

    baseline = json.loads(open(args.baseline, encoding="utf-8").read())
    candidate = json.loads(open(args.candidate, encoding="utf-8").read())
    report = compare_promotion_benchmarks(
        baseline,
        candidate,
        latency_tolerance=args.latency_tolerance,
        rss_tolerance=args.rss_tolerance,
        candidate_recall_tolerance=args.candidate_recall_tolerance,
    )
    if args.output:
        _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["promote"] else 1


def _write_report(path: str, report: dict[str, object]) -> None:
    """Write deterministic JSON; CI can upload the file without tracking it."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
