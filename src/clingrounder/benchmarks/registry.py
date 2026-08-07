"""Discovery and lazy dispatch for benchmark plugins."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import cast

from clingrounder.benchmarks.plugins import BenchmarkHandler, BenchmarkPlugin

__all__ = [
    "BENCHMARK_ENTRY_POINT_GROUP",
    "benchmark_plugins",
    "resolve_benchmark_handler",
]

BENCHMARK_ENTRY_POINT_GROUP = "clingrounder.benchmarks"


def benchmark_plugins() -> tuple[BenchmarkPlugin, ...]:
    """Load built-in and installed plugins in deterministic name order."""

    from clingrounder.benchmarks.phase1.plugin import PHASE1_PLUGIN

    plugins: dict[str, BenchmarkPlugin] = {
        PHASE1_PLUGIN.name: cast(BenchmarkPlugin, PHASE1_PLUGIN)
    }
    for entry_point in entry_points(group=BENCHMARK_ENTRY_POINT_GROUP):
        plugin = entry_point.load()
        if not _is_plugin(plugin):
            raise TypeError(f"Benchmark entry point {entry_point.name!r} is not a plugin")
        if plugin.name in plugins:
            raise ValueError(f"Duplicate benchmark plugin name: {plugin.name}")
        plugins[plugin.name] = plugin
    return tuple(plugins[name] for name in sorted(plugins))


def resolve_benchmark_handler(handler_name: str) -> BenchmarkHandler | None:
    """Resolve one benchmark-owned handler without importing its implementation module."""

    for plugin in benchmark_plugins():
        handler = plugin.handlers().get(handler_name)
        if handler is not None:
            return handler
    return None


def _is_plugin(value: object) -> bool:
    return (
        isinstance(getattr(value, "name", None), str)
        and isinstance(getattr(value, "summary", None), str)
        and callable(getattr(value, "register_cli", None))
        and callable(getattr(value, "handlers", None))
    )
