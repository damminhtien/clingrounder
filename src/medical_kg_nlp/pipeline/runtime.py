"""Runtime capability contracts for concurrent pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["DeviceKind", "RuntimeCapabilities"]

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
