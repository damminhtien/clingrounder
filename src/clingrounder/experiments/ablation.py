"""Ablation result records and CSV renderers built on neutral runtime metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.evaluation.runtime_metrics import (
    ScalarMetric,
    StageAggregate,
    aggregate_traces,
    flatten_metrics,
)

__all__ = [
    "AblationVariantResult",
    "StageAggregate",
    "aggregate_traces",
    "flatten_metrics",
    "write_stage_timings_csv",
    "write_summary_csv",
]


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
