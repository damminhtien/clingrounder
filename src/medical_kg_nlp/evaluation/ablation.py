from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medical_kg_nlp.pipeline.tracing import PipelineTrace


ScalarMetric = float | int | str | None


@dataclass(frozen=True)
class StageAggregate:
    stage: str
    calls: int
    total_ms: float
    avg_ms: float
    max_ms: float
    counters: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "calls": self.calls,
            "total_ms": round(self.total_ms, 6),
            "avg_ms": round(self.avg_ms, 6),
            "max_ms": round(self.max_ms, 6),
            "counters": self.counters,
        }


@dataclass(frozen=True)
class AblationVariantResult:
    name: str
    metrics: dict[str, Any]
    stage_aggregates: list[StageAggregate]
    prediction_path: str
    trace_path: str
    total_ms: float
    docs_per_second: float
    validation_issues: int

    @property
    def bottleneck_stage(self) -> StageAggregate | None:
        if not self.stage_aggregates:
            return None
        return max(self.stage_aggregates, key=lambda stage: stage.total_ms)

    def to_json(self) -> dict[str, Any]:
        bottleneck = self.bottleneck_stage
        return {
            "name": self.name,
            "metrics": self.metrics,
            "stages": [stage.to_json() for stage in self.stage_aggregates],
            "prediction_path": self.prediction_path,
            "trace_path": self.trace_path,
            "total_ms": round(self.total_ms, 6),
            "docs_per_second": round(self.docs_per_second, 6),
            "validation_issues": self.validation_issues,
            "bottleneck_stage": bottleneck.stage if bottleneck else None,
        }


def aggregate_traces(traces: list[PipelineTrace]) -> list[StageAggregate]:
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
    flattened: dict[str, ScalarMetric] = {}
    _flatten_mapping(metrics, "", flattened)
    return flattened


def write_summary_csv(path: str | Path, results: list[AblationVariantResult]) -> None:
    metric_keys = sorted({key for result in results for key in flatten_metrics(result.metrics)})
    fieldnames = [
        "variant",
        "total_ms",
        "docs_per_second",
        "bottleneck_stage",
        "bottleneck_total_ms",
        "validation_issues",
        "prediction_path",
        "trace_path",
        *[f"metric_{key}" for key in metric_keys],
    ]

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            bottleneck = result.bottleneck_stage
            row: dict[str, ScalarMetric] = {
                "variant": result.name,
                "total_ms": round(result.total_ms, 6),
                "docs_per_second": round(result.docs_per_second, 6),
                "bottleneck_stage": bottleneck.stage if bottleneck else None,
                "bottleneck_total_ms": round(bottleneck.total_ms, 6) if bottleneck else None,
                "validation_issues": result.validation_issues,
                "prediction_path": result.prediction_path,
                "trace_path": result.trace_path,
            }
            row.update({f"metric_{key}": value for key, value in flatten_metrics(result.metrics).items()})
            writer.writerow(row)


def write_stage_timings_csv(path: str | Path, results: list[AblationVariantResult]) -> None:
    counter_keys = sorted(
        {
            key
            for result in results
            for stage in result.stage_aggregates
            for key in stage.counters
        }
    )
    fieldnames = [
        "variant",
        "stage",
        "calls",
        "total_ms",
        "avg_ms",
        "max_ms",
        *[f"counter_{key}" for key in counter_keys],
    ]

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for stage in result.stage_aggregates:
                row: dict[str, ScalarMetric] = {
                    "variant": result.name,
                    "stage": stage.stage,
                    "calls": stage.calls,
                    "total_ms": round(stage.total_ms, 6),
                    "avg_ms": round(stage.avg_ms, 6),
                    "max_ms": round(stage.max_ms, 6),
                }
                row.update({f"counter_{key}": stage.counters.get(key, 0) for key in counter_keys})
                writer.writerow(row)


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
        elif isinstance(value, (int, float, str)) or value is None:
            output[name] = value
        else:
            output[name] = json.dumps(value, ensure_ascii=False, sort_keys=True)
