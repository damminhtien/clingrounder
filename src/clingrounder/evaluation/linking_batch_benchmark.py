"""Measured scalar-versus-batch candidate reranking benchmarks."""

from __future__ import annotations

import resource
import sys
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Any

from clingrounder.linking.batch import CandidateRerankRequest

__all__ = ["CandidateBatchBenchmarkReport", "benchmark_candidate_reranker"]


@dataclass(frozen=True)
class CandidateBatchBenchmarkReport:
    """Measured workload metrics and exact output-equivalence result."""

    request_count: int
    pair_count: int
    repetitions: int
    scalar_median_ms: float
    scalar_p95_ms: float
    batch_median_ms: float
    batch_p95_ms: float
    peak_rss_mb: float
    scalar_tokenizer_calls: int
    scalar_model_forward_passes: int
    batch_tokenizer_calls: int
    batch_model_forward_passes: int
    output_equivalent: bool


def benchmark_candidate_reranker(
    reranker: Any,
    requests: tuple[CandidateRerankRequest, ...],
    *,
    repetitions: int = 3,
) -> CandidateBatchBenchmarkReport:
    """Run scalar and batch paths and report measured latency and model-work counters."""

    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    pair_count = sum(len(request.candidates) for request in requests)
    scalar_times: list[float] = []
    batch_times: list[float] = []
    scalar_outputs: dict[str, list[Any]] | None = None
    batch_outputs: dict[str, list[Any]] | None = None
    scalar_stats_before = _stats(reranker)
    for _ in range(repetitions):
        started = perf_counter()
        scalar_outputs = {
            request.entity_id: reranker.rerank(
                list(request.candidates), request.context_window, request.mention
            )
            for request in requests
        }
        scalar_times.append((perf_counter() - started) * 1000.0)
    scalar_stats_after = _stats(reranker)
    for _ in range(repetitions):
        started = perf_counter()
        batch_outputs = reranker.rerank_batch(requests)
        batch_times.append((perf_counter() - started) * 1000.0)
    batch_stats_after = _stats(reranker)
    return CandidateBatchBenchmarkReport(
        request_count=len(requests),
        pair_count=pair_count,
        repetitions=repetitions,
        scalar_median_ms=median(scalar_times),
        scalar_p95_ms=_p95(scalar_times),
        batch_median_ms=median(batch_times),
        batch_p95_ms=_p95(batch_times),
        peak_rss_mb=_peak_rss_mb(),
        scalar_tokenizer_calls=scalar_stats_after[0] - scalar_stats_before[0],
        scalar_model_forward_passes=scalar_stats_after[1] - scalar_stats_before[1],
        batch_tokenizer_calls=batch_stats_after[0] - scalar_stats_after[0],
        batch_model_forward_passes=batch_stats_after[1] - scalar_stats_after[1],
        output_equivalent=scalar_outputs == batch_outputs,
    )


def _stats(reranker: Any) -> tuple[int, int]:
    stats = reranker.stats() if callable(getattr(reranker, "stats", None)) else {}
    return int(stats.get("tokenizer_calls", 0)), int(stats.get("model_forward_passes", 0))


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[index]


def _peak_rss_mb() -> float:
    """Normalize platform-specific ``ru_maxrss`` units for comparable reports."""

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # INVARIANT: macOS reports bytes while Linux reports kibibytes.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor
