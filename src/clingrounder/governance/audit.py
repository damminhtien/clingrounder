"""Bounded audit events that never store raw clinical text."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

__all__ = ["AuditEvent", "AuditSink", "InMemoryAuditSink", "NoOpAuditSink"]


class AuditSink(Protocol):
    def emit(self, event: "AuditEvent") -> None: ...


class NoOpAuditSink:
    """Default sink; audit integration never changes prediction behavior."""

    def emit(self, event: "AuditEvent") -> None:
        del event


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    outcome: str = "success"
    document_id_hash: str | None = None
    artifact_sha256: str | None = None
    profile_fingerprint: str | None = None
    model_revision: str | None = None
    terminology_fingerprint: str | None = None
    details: Mapping[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        forbidden = {"text", "raw_text", "mention", "content", "note", "document"}
        if forbidden.intersection(self.details):
            raise ValueError("Audit details cannot contain raw clinical text fields")
        if len(self.details) > 24:
            raise ValueError("Audit details are bounded to 24 fields")

    def to_json(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "outcome": self.outcome,
            "document_id_hash": self.document_id_hash,
            "artifact_sha256": self.artifact_sha256,
            "profile_fingerprint": self.profile_fingerprint,
            "model_revision": self.model_revision,
            "terminology_fingerprint": self.terminology_fingerprint,
            "details": dict(self.details),
            "timestamp": self.timestamp,
        }


class InMemoryAuditSink:
    """Thread-safe bounded sink useful for tests and local research traces."""

    def __init__(self, *, max_events: int = 4096) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self._max_events = max_events
        self._lock = threading.Lock()
        self.events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        with self._lock:
            self.events.append(event)
            del self.events[:-self._max_events]

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [event.to_json() for event in self.events]

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(row, sort_keys=True) for row in self.snapshot())
