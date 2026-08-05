"""Runtime capability and lifecycle contracts for pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from medical_kg_nlp.pipeline.runner import PipelineRunner

__all__ = ["Closable", "DeviceKind", "PipelineRuntime", "RuntimeCapabilities"]


class Closable(Protocol):
    """Small lifecycle contract for external or expensive runtime resources."""

    def close(self) -> None: ...

DeviceKind = Literal["cpu", "cuda", "mps", "other"]


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Declare how one composed pipeline may be used by a batch executor."""

    thread_safe: bool = False
    process_safe: bool = True
    supports_batching: bool = False
    device_kind: DeviceKind = "cpu"

    def __post_init__(self) -> None:
        if self.device_kind not in {"cpu", "cuda", "mps", "other"}:
            raise ValueError(f"Unsupported device kind {self.device_kind!r}")


class PipelineRuntime:
    """Own a composed runner and close its resources deterministically.

    Resources are supplied in composition order and closed in reverse order. The owner is
    intentionally lightweight: it does not introduce a service container or background loop.
    """

    def __init__(self, runner: "PipelineRunner", resources: Iterable[Closable] = ()) -> None:
        self._runner = runner
        self._resources = tuple(resources)
        self._closed = False

    @property
    def runner(self) -> "PipelineRunner":
        """Return the runner while allowing callers to manage its owner explicitly."""

        if self._closed:
            raise RuntimeError("PipelineRuntime is closed")
        return self._runner

    def __enter__(self) -> "PipelineRuntime":
        if self._closed:
            raise RuntimeError("PipelineRuntime is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Close each unique resource once; safe after partial initialization and exceptions."""

        if self._closed:
            return
        self._closed = True
        seen: set[int] = set()
        failures: list[BaseException] = []
        for resource in reversed((self._runner, *self._resources)):
            identity = id(resource)
            if identity in seen:
                continue
            seen.add(identity)
            try:
                resource.close()
            except BaseException as error:  # pragma: no cover - defensive cleanup path.
                failures.append(error)
        if failures:
            raise RuntimeError("One or more pipeline resources failed to close") from failures[0]
