"""Portable process-memory measurements used by benchmark reports."""

from __future__ import annotations

import sys

try:  # ``resource`` is unavailable on Windows.
    import resource
except ImportError:  # pragma: no cover - exercised only on Windows.
    resource = None  # type: ignore[assignment]

__all__ = ["peak_rss_bytes", "peak_rss_mb", "rss_bytes_from_ru_maxrss"]


def peak_rss_bytes() -> int:
    """Return the process peak resident set size in bytes, or zero if unsupported."""

    if resource is None:
        return 0
    return rss_bytes_from_ru_maxrss(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        platform=sys.platform,
    )


def peak_rss_mb() -> float:
    """Return :func:`peak_rss_bytes` converted to mebibytes."""

    return peak_rss_bytes() / (1024.0 * 1024.0)


def rss_bytes_from_ru_maxrss(value: int | float, *, platform: str) -> int:
    """Normalize ``ru_maxrss`` to bytes for a named platform.

    ``resource.getrusage`` reports bytes on macOS and kibibytes on Linux and the BSDs used by
    the supported Unix runners.  Keeping this conversion in one helper prevents benchmark
    modules from silently disagreeing about memory units.
    """

    if value < 0:
        raise ValueError("ru_maxrss cannot be negative")
    # INVARIANT: macOS already reports bytes; multiplying it by 1024 overstates RSS by 1024x.
    return int(value) if platform == "darwin" else int(value) * 1024
