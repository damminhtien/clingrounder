"""Task-neutral benchmark plugin commands."""

from __future__ import annotations

import argparse
import json

from clingrounder.benchmarks.registry import benchmark_plugins

from clingrounder.evaluation.promotion_benchmark import (
    compare_promotion_benchmarks,
    run_promotion_benchmark,
)
from clingrounder.evaluation.dataset_benchmark import run_dataset_benchmark
from clingrounder.evaluation.dataset_benchmark import run_dataset_benchmark_suite
from clingrounder.evaluation.dataset_audit import audit_dataset

__all__ = [
    "compare_runtime_benchmark",
    "list_benchmarks",
    "run_runtime_benchmark",
    "run_dataset_benchmark_command",
    "run_dataset_benchmark_suite_command",
    "audit_dataset_command",
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


def run_dataset_benchmark_command(args: argparse.Namespace) -> int:
    """Run a benchmark contract directory and print its measured summary."""

    report = run_dataset_benchmark(
        args.benchmark,
        args.config,
        args.output,
        split=args.split,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_dataset_benchmark_suite_command(args: argparse.Namespace) -> int:
    """Run named product profiles and write one reproducible ablation index."""

    configs: dict[str, str] = {}
    for value in args.config:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("--config must use NAME=PATH")
        if name in configs:
            raise ValueError(f"Duplicate benchmark suite config name: {name!r}")
        configs[name] = path
    report = run_dataset_benchmark_suite(
        args.benchmark,
        configs,
        args.output,
        split=args.split,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def audit_dataset_command(args: argparse.Namespace) -> int:
    """Audit split leakage and review provenance before public benchmark use."""

    report = audit_dataset(args.benchmark)
    payload = report.to_dict()
    if args.output:
        _write_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.strict and not report.eligible_for_clinical_claim:
        return 1
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
