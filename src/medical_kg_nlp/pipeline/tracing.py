from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator


@dataclass(frozen=True)
class StageMeasurement:
    name: str
    elapsed_ms: float
    counters: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_ms": round(self.elapsed_ms, 6),
            "counters": self.counters,
        }


@dataclass
class PipelineTrace:
    document_id: str
    stages: list[StageMeasurement] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, int]]:
        counters: dict[str, int] = {}
        start = perf_counter()
        try:
            yield counters
        finally:
            elapsed_ms = (perf_counter() - start) * 1000
            self.stages.append(StageMeasurement(name=name, elapsed_ms=elapsed_ms, counters=counters))

    @property
    def total_ms(self) -> float:
        return sum(stage.elapsed_ms for stage in self.stages)

    def bottleneck(self) -> StageMeasurement | None:
        if not self.stages:
            return None
        return max(self.stages, key=lambda stage: stage.elapsed_ms)

    def to_json(self) -> dict[str, Any]:
        bottleneck = self.bottleneck()
        return {
            "document_id": self.document_id,
            "total_ms": round(self.total_ms, 6),
            "bottleneck_stage": bottleneck.name if bottleneck else None,
            "stages": [stage.to_json() for stage in self.stages],
        }
