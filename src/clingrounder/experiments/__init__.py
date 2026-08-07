"""Ablation orchestration, experiment memory, and agent-facing loop reports."""

from clingrounder.experiments.ablation import (
    AblationVariantResult,
    StageAggregate,
    aggregate_traces,
    flatten_metrics,
)
from clingrounder.experiments.loop_engineer import (
    build_loop_engineering_report,
    write_loop_engineering_report,
)

__all__ = [
    "AblationVariantResult",
    "StageAggregate",
    "aggregate_traces",
    "build_loop_engineering_report",
    "flatten_metrics",
    "write_loop_engineering_report",
]
