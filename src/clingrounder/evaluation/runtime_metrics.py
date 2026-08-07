"""Task-neutral aggregation of pipeline traces and nested scalar metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from clingrounder.pipeline.tracing import PipelineTrace

__all__ = ["ScalarMetric", "StageAggregate", "aggregate_traces", "flatten_metrics"]

ScalarMetric = float | int | str | None


@dataclass(frozen=True)
class StageAggregate:
    """Runtime totals and counters for one stable pipeline stage name."""

    stage: str
    calls: int
    total_ms: float
    avg_ms: float
    max_ms: float
    counters: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return deterministic machine-readable timing fields."""

        return {
            "stage": self.stage,
            "calls": self.calls,
            "total_ms": round(self.total_ms, 6),
            "avg_ms": round(self.avg_ms, 6),
            "max_ms": round(self.max_ms, 6),
            "counters": self.counters,
        }


def aggregate_traces(traces: list[PipelineTrace]) -> list[StageAggregate]:
    """Aggregate traces while preserving first-observed stage order."""

    stage_order: list[str] = []
    elapsed_by_stage: dict[str, list[float]] = defaultdict(list)
    counters_by_stage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for trace in traces:
        for stage in trace.stages:
            if stage.name not in elapsed_by_stage:
                stage_order.append(stage.name)
            elapsed_by_stage[stage.name].append(stage.elapsed_ms)
            for key, value in stage.counters.items():
                counters_by_stage[stage.name][key] += value

    aggregates: list[StageAggregate] = []
    for stage_name in stage_order:
        values = elapsed_by_stage[stage_name]
        total_ms = sum(values)
        aggregates.append(
            StageAggregate(
                stage=stage_name,
                calls=len(values),
                total_ms=total_ms,
                avg_ms=total_ms / len(values),
                max_ms=max(values),
                counters=dict(counters_by_stage[stage_name]),
            )
        )
    return aggregates


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, ScalarMetric]:
    """Flatten nested metric mappings for stable CSV column generation."""

    flattened: dict[str, ScalarMetric] = {}
    _flatten_mapping(metrics, "", flattened)
    return flattened


def _flatten_mapping(
    payload: dict[str, Any],
    prefix: str,
    output: dict[str, ScalarMetric],
) -> None:
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_mapping(value, name, output)
        elif isinstance(value, bool):
            output[name] = int(value)
        elif isinstance(value, int | float | str) or value is None:
            output[name] = value
        else:
            output[name] = json.dumps(value, ensure_ascii=False, sort_keys=True)
