"""Task-neutral benchmark plugin commands."""

from __future__ import annotations

import argparse
import json

from medical_kg_nlp.benchmarks.registry import benchmark_plugins

__all__ = ["list_benchmarks"]


def list_benchmarks(args: argparse.Namespace) -> int:
    """List installed benchmark plugins without importing benchmark implementations."""

    del args
    payload = [
        {"name": plugin.name, "summary": plugin.summary}
        for plugin in benchmark_plugins()
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
