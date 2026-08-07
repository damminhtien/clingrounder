"""PHI-safe pipeline tracing and vendor-neutral observer ports."""

from __future__ import annotations

import hashlib
import importlib
import threading
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterator, Literal, Protocol

__all__ = [
    "InMemoryPipelineObserver",
    "NoOpPipelineObserver",
    "OpenTelemetryPipelineObserver",
    "PipelineObserverPort",
    "PipelineTrace",
    "StageMeasurement",
]

StageStatus = Literal["success", "failure", "cancelled"]
_MAX_ERROR_MESSAGE = 512
_MAX_TRACE_STAGES = 128


class PipelineObserverPort(Protocol):
    """Minimal observer contract; implementations must not influence inference."""

    def stage_started(self, stage: str, document_id: str, metadata: Mapping[str, Any]) -> None: ...

    def stage_completed(self, measurement: "StageMeasurement") -> None: ...

    def stage_failed(self, measurement: "StageMeasurement", error: BaseException) -> None: ...

    def counter(self, name: str, value: int = 1, labels: Mapping[str, str] | None = None) -> None: ...


class NoOpPipelineObserver:
    """Default observer with no allocations beyond the trace itself."""

    def stage_started(self, stage: str, document_id: str, metadata: Mapping[str, Any]) -> None:
        return None

    def stage_completed(self, measurement: "StageMeasurement") -> None:
        return None

    def stage_failed(self, measurement: "StageMeasurement", error: BaseException) -> None:
        return None

    def counter(self, name: str, value: int = 1, labels: Mapping[str, str] | None = None) -> None:
        return None


@dataclass(frozen=True)
class StageMeasurement:
    """Bounded stage result retaining the old positional constructor."""

    name: str
    elapsed_ms: float
    counters: dict[str, int] = field(default_factory=dict)
    status: StageStatus = "success"
    started_at: str = ""
    error_type: str | None = None
    error_message: str | None = None
    document_length: int | None = None
    entity_count: int | None = None
    configuration_fingerprint: str | None = None
    terminology_fingerprint: str | None = None
    model_revision: str | None = None
    backend: str | None = None
    worker: str | None = None
    queue_wait_ms: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_ms": round(self.elapsed_ms, 6),
            "status": self.status,
            "started_at": self.started_at,
            "counters": dict(self.counters),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "document_length": self.document_length,
            "entity_count": self.entity_count,
            "configuration_fingerprint": self.configuration_fingerprint,
            "terminology_fingerprint": self.terminology_fingerprint,
            "model_revision": self.model_revision,
            "backend": self.backend,
            "worker": self.worker,
            "queue_wait_ms": round(self.queue_wait_ms, 6),
        }


class InMemoryPipelineObserver:
    """Thread-safe research observer with bounded, PHI-safe aggregate metrics."""

    def __init__(self, *, retain_stages: int = 4096) -> None:
        if retain_stages < 1:
            raise ValueError("retain_stages must be positive")
        self._lock = threading.Lock()
        self._retain_stages = retain_stages
        self.stages: list[StageMeasurement] = []
        self.counters: Counter[str] = Counter()
        self.documents_processed = 0
        self.documents_failed = 0

    def stage_started(self, stage: str, document_id: str, metadata: Mapping[str, Any]) -> None:
        return None

    def stage_completed(self, measurement: StageMeasurement) -> None:
        with self._lock:
            self.stages.append(measurement)
            del self.stages[:-self._retain_stages]
            for name, value in measurement.counters.items():
                self.counters[f"stage.{measurement.name}.{name}"] += value

    def stage_failed(self, measurement: StageMeasurement, error: BaseException) -> None:
        self.stage_completed(measurement)

    def counter(self, name: str, value: int = 1, labels: Mapping[str, str] | None = None) -> None:
        # High-cardinality labels are intentionally ignored by the in-memory aggregate.
        with self._lock:
            self.counters[name] += value

    def record_document(self, *, success: bool, entities: int, assigned_codes: int) -> None:
        with self._lock:
            if success:
                self.documents_processed += 1
            else:
                self.documents_failed += 1
            self.counters["entities"] += entities
            self.counters["assigned_codes"] += assigned_codes

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            entities = self.counters.get("entities", 0)
            assigned_codes = self.counters.get("assigned_codes", 0)
            candidate_histogram = {
                key.removeprefix("stage.candidate_generation."): value
                for key, value in self.counters.items()
                if key.startswith("stage.candidate_generation.candidate_count_")
            }
            validation_failures = {
                key.removeprefix("stage.prediction_validation.validation_"): value
                for key, value in self.counters.items()
                if key.startswith("stage.prediction_validation.validation_")
            }
            model_forward_passes = sum(
                value
                for key, value in self.counters.items()
                if key.endswith("model_forward_passes")
            )
            return {
                "documents_processed": self.documents_processed,
                "documents_failed": self.documents_failed,
                "counters": dict(self.counters),
                "assigned_code_coverage": assigned_codes / entities if entities else 0.0,
                "candidate_count_distribution": candidate_histogram,
                "validation_failures_by_type": validation_failures,
                "model_forward_pass_count": model_forward_passes,
                "abstention_rate": (
                    1.0 - assigned_codes / entities if entities else 0.0
                ),
                "stages": [stage.to_json() for stage in self.stages],
            }


class OpenTelemetryPipelineObserver:
    """Optional OpenTelemetry bridge loaded only when explicitly instantiated."""

    def __init__(self, tracer: Any = None, meter: Any = None) -> None:
        try:
            metrics = importlib.import_module("opentelemetry.metrics")
            trace = importlib.import_module("opentelemetry.trace")
        except ImportError as error:  # pragma: no cover - depends on optional extra.
            raise RuntimeError(
                "OpenTelemetry support requires the optional observability dependencies"
            ) from error
        self._tracer = tracer or trace.get_tracer("clingrounder")
        self._meter = meter or metrics.get_meter("clingrounder")
        self._stage_latency = self._meter.create_histogram("clingrounder.pipeline.stage_ms")
        self._counter = self._meter.create_counter("clingrounder.pipeline.events")

    def stage_started(self, stage: str, document_id: str, metadata: Mapping[str, Any]) -> None:
        return None

    def stage_completed(self, measurement: StageMeasurement) -> None:
        self._stage_latency.record(
            measurement.elapsed_ms,
            {"stage": measurement.name, "status": measurement.status},
        )

    def stage_failed(self, measurement: StageMeasurement, error: BaseException) -> None:
        self.stage_completed(measurement)
        self._counter.add(1, {"event": "stage_failure", "stage": measurement.name})

    def counter(self, name: str, value: int = 1, labels: Mapping[str, str] | None = None) -> None:
        safe_labels = {key: value for key, value in (labels or {}).items() if key != "document_id"}
        safe_labels["event"] = name
        self._counter.add(value, safe_labels)


@dataclass
class PipelineTrace:
    document_id: str
    observer: PipelineObserverPort | None = None
    document_length: int | None = None
    configuration_fingerprint: str | None = None
    terminology_fingerprint: str | None = None
    model_revision: str | None = None
    backend: str | None = None
    worker: str | None = None
    queue_wait_ms: float = 0.0
    redact_errors: bool = True
    stages: list[StageMeasurement] = field(default_factory=list)
    _observer: PipelineObserverPort = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._observer = self.observer or NoOpPipelineObserver()
        _safe_observe(self._observer.counter, "documents_started")

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, int]]:
        counters: dict[str, int] = {}
        started_at = datetime.now(timezone.utc).isoformat()
        _safe_observe(
            self._observer.stage_started,
            name,
            _hash_identifier(self.document_id),
            {
                "configuration_fingerprint": self.configuration_fingerprint,
                "terminology_fingerprint": self.terminology_fingerprint,
                "model_revision": self.model_revision,
                "backend": self.backend,
                "worker": self.worker,
            },
        )
        start = perf_counter()
        try:
            yield counters
        except BaseException as error:
            measurement = self._measurement(
                name, counters, started_at, start, error=error
            )
            self._append(measurement)
            _safe_observe(self._observer.stage_failed, measurement, error)
            raise
        else:
            measurement = self._measurement(name, counters, started_at, start)
            self._append(measurement)
            _safe_observe(self._observer.stage_completed, measurement)

    def _measurement(
        self,
        name: str,
        counters: dict[str, int],
        started_at: str,
        start: float,
        error: BaseException | None = None,
    ) -> StageMeasurement:
        status: StageStatus = "success"
        if error is not None:
            status = "cancelled" if error.__class__.__name__ == "CancelledError" else "failure"
        return StageMeasurement(
            name=name,
            elapsed_ms=(perf_counter() - start) * 1000,
            counters=counters,
            status=status,
            started_at=started_at,
            error_type=None if error is None else type(error).__name__,
            error_message=(
                None
                if error is None
                else ("redacted" if self.redact_errors else _bounded_error(error))
            ),
            document_length=self.document_length,
            entity_count=counters.get("entities"),
            configuration_fingerprint=self.configuration_fingerprint,
            terminology_fingerprint=self.terminology_fingerprint,
            model_revision=self.model_revision,
            backend=self.backend,
            worker=self.worker,
            queue_wait_ms=self.queue_wait_ms,
        )

    def _append(self, measurement: StageMeasurement) -> None:
        if len(self.stages) >= _MAX_TRACE_STAGES:
            return
        self.stages.append(measurement)

    def mark_finished(self, *, success: bool, entities: int = 0, assigned_codes: int = 0) -> None:
        _safe_observe(self._observer.counter, "documents_processed" if success else "documents_failed")
        record_document = getattr(self._observer, "record_document", None)
        if callable(record_document):
            record_document(success=success, entities=entities, assigned_codes=assigned_codes)

    def attach_to(self, error: BaseException) -> None:
        """Expose a partial trace without changing the original exception type."""

        try:
            setattr(error, "pipeline_trace", self)
        except Exception:
            return None

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
            "document_id_hash": _hash_identifier(self.document_id),
            "total_ms": round(self.total_ms, 6),
            "bottleneck_stage": bottleneck.name if bottleneck else None,
            "stages": [stage.to_json() for stage in self.stages],
        }


def _hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _bounded_error(error: BaseException) -> str:
    message = str(error).replace("\n", " ").strip()
    return message[:_MAX_ERROR_MESSAGE]


def _safe_observe(callback: Any, *args: Any) -> None:
    """Observability failures must never change prediction or error semantics."""

    try:
        callback(*args)
    except Exception:
        return None
