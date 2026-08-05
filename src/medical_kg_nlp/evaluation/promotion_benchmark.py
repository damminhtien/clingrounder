"""Reproducible, task-neutral promotion benchmarks for pipeline profiles.

The benchmark deliberately separates correctness gates from machine-sensitive
timings.  Fixture gold is optional: when a fixture does not define a metric,
the report records ``null`` and comparison does not invent a passing score.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

try:  # ``resource`` is available on Unix, but not on every supported platform.
    import resource
except ImportError:  # pragma: no cover - Windows only.
    resource = None  # type: ignore[assignment]

from medical_kg_nlp.pipeline.factory import PipelineFactory
from medical_kg_nlp.pipeline.config_loader import ResolvedPipelineConfig
from medical_kg_nlp.pipeline.runner import PipelineRunner
from medical_kg_nlp.pipeline.tracing import InMemoryPipelineObserver
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.validator import PredictionValidator

__all__ = ["compare_promotion_benchmarks", "run_promotion_benchmark"]


@dataclass(frozen=True)
class BenchmarkInput:
    """One redistributable document and optional, text-addressed gold facts."""

    document_id: str
    text: str
    metadata: dict[str, str]
    gold_candidates: tuple[tuple[str, tuple[str, ...]], ...] = ()
    gold_assertions: tuple[tuple[str, tuple[str, ...]], ...] = ()


def run_promotion_benchmark(
    input_path: str | Path,
    config_path: str | Path,
    *,
    repeats: int = 5,
    warmup: int = 1,
    candidate_ks: tuple[int, ...] = (1, 5, 20),
) -> dict[str, Any]:
    """Run warm-ups and repeated passes, returning a JSON-safe CI artifact."""

    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup must be non-negative")
    resolved = ResolvedPipelineConfig.load(config_path, require_profile=True)
    inputs = _load_inputs(input_path)
    init_start = perf_counter()
    base_runner = PipelineFactory.from_config(resolved.factory_config)
    observer = InMemoryPipelineObserver()
    # The observer belongs to this benchmark run.  ``replace`` preserves every
    # immutable component and avoids mutating a runner shared by another job.
    runner = PipelineRunner(replace(base_runner.components, observer=observer))
    initialization_ms = (perf_counter() - init_start) * 1000

    repeat_results: list[list[tuple[BenchmarkInput, ClinicalPrediction]]] = []
    timings: list[float] = []
    rss_before = _peak_rss_bytes()
    try:
        for _ in range(warmup):
            _run_once(runner, inputs)
        for _ in range(repeats):
            started = perf_counter()
            repeat_results.append(_run_once(runner, inputs))
            timings.append((perf_counter() - started) * 1000)
        rss_after = _peak_rss_bytes()
        correctness = _correctness_report(
            repeat_results,
            runner,
            candidate_ks,
        )
        stage_latency = _stage_latency(observer.snapshot())
        last_results = repeat_results[-1]
        total_documents = len(inputs)
        median_ms = statistics.median(timings)
        snapshot = observer.snapshot()
        return {
            "schema_version": "medical-kg.promotion-benchmark.v1",
            "benchmark": {
                "input": str(Path(input_path).resolve()),
                "config": str(Path(config_path).resolve()),
                "repeats": repeats,
                "warmup": warmup,
                "documents": total_documents,
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "commit": _git_commit(),
            },
            "profile": {
                "profile_sha256": resolved.inspection_report()["profile_sha256"],
                "configuration_fingerprint": runner.components.configuration_fingerprint,
                "terminology_fingerprint": runner.components.terminology_fingerprint,
                "model_revision": runner.components.model_revision,
            },
            "performance": {
                "initialization_ms": round(initialization_ms, 6),
                "documents_per_second": round(total_documents / (median_ms / 1000), 6),
                "entities_per_second": round(
                    correctness["entity_count"] / (median_ms / 1000), 6
                ),
                "document_latency_ms": _percentiles(timings),
                "peak_rss_bytes": max(rss_before, rss_after),
                "terminology_lookup_latency_ms": stage_latency.get(
                    "candidate_generation", {}
                ),
                "model_forward_pass_count": snapshot["counters"].get(
                    "stage.candidate_reranking.model_forward_passes", 0
                ),
            },
            "correctness": correctness,
            "stage_latency": stage_latency,
            "trace_summary": {
                "documents_processed": snapshot["documents_processed"],
                "documents_failed": snapshot["documents_failed"],
                "validation_error_count": correctness["validation_error_count"],
            },
            "reproducibility": {
                "fixed_profile": True,
                "warmup_excluded": True,
                "timing_unit": "milliseconds",
                "raw_text_emitted": False,
                "document_outputs_hashed": True,
                "last_run_documents": len(last_results),
            },
        }
    finally:
        runner.close()
        base_runner.close()


def compare_promotion_benchmarks(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    latency_tolerance: float = 0.10,
    rss_tolerance: float = 0.20,
    candidate_recall_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare two reports with hard correctness and tolerant performance gates."""

    base_c = baseline["correctness"]
    cand_c = candidate["correctness"]
    base_p = baseline["performance"]
    cand_p = candidate["performance"]
    recall_gates = {
        str(k): _protected_metric(
            cand_c["candidate_recall_at_k"].get(str(k)),
            base_c["candidate_recall_at_k"].get(str(k)),
            candidate_recall_tolerance,
        )
        for k in (1, 5, 20)
    }
    gates: dict[str, bool | None] = {
        "offset_validity": cand_c["offset_validity"] == 1.0,
        "invalid_assigned_code_rate": cand_c["invalid_assigned_code_rate"] == 0.0,
        "invalid_relation_rate": cand_c["invalid_relation_rate"] == 0.0,
        "deterministic_output": cand_c["deterministic_output"],
        "candidate_ordering_stable": cand_c["candidate_ordering_stable"],
        "candidate_recall_at_k": all(recall_gates.values()),
        "assertion_positive_recall": _protected_metric(
            cand_c.get("assertion_positive_recall"),
            base_c.get("assertion_positive_recall"),
            candidate_recall_tolerance,
        ),
        "latency_within_tolerance": cand_p["document_latency_ms"]["p50"]
        <= base_p["document_latency_ms"]["p50"] * (1 + latency_tolerance),
        "rss_within_tolerance": cand_p["peak_rss_bytes"]
        <= base_p["peak_rss_bytes"] * (1 + rss_tolerance),
    }
    # A missing protected metric is an explicit review gate, not an accidental pass.
    promotable = all(value is True for value in gates.values())
    return {
        "schema_version": "medical-kg.promotion-comparison.v1",
        "promote": promotable,
        "gates": gates,
        "candidate_recall_gates": recall_gates,
        "baseline_commit": baseline.get("environment", {}).get("commit"),
        "candidate_commit": candidate.get("environment", {}).get("commit"),
    }


def _load_inputs(path: str | Path) -> list[BenchmarkInput]:
    inputs: list[BenchmarkInput] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid benchmark JSONL at line {line_number}") from error
        inputs.append(
            BenchmarkInput(
                document_id=str(row["document_id"]),
                text=str(row["text"]),
                metadata={str(k): str(v) for k, v in row.get("metadata", {}).items()},
                gold_candidates=tuple(
                    (str(item["text"]), tuple(str(code) for code in item.get("codes", [])))
                    for item in row.get("gold_candidates", [])
                ),
                gold_assertions=tuple(
                    (str(item["text"]), tuple(str(value) for value in item["assertions"]))
                    for item in row.get("gold_assertions", [])
                ),
            )
        )
    if not inputs:
        raise ValueError(f"Benchmark input {path} contains no documents")
    return inputs


def _run_once(
    runner: PipelineRunner,
    inputs: list[BenchmarkInput],
) -> list[tuple[BenchmarkInput, ClinicalPrediction]]:
    return [
        (item, runner.process_text_with_trace(item.document_id, item.text, item.metadata).prediction)
        for item in inputs
    ]


def _correctness_report(
    repeat_results: list[list[tuple[BenchmarkInput, ClinicalPrediction]]],
    runner: PipelineRunner,
    candidate_ks: tuple[int, ...],
) -> dict[str, Any]:
    results = repeat_results[-1]
    entity_count = sum(len(prediction.entities) for _, prediction in results)
    assigned = sum(
        entity.code is not None for _, prediction in results for entity in prediction.entities
    )
    invalid_codes = 0
    invalid_relations = 0
    validation_errors = 0
    offset_valid = True
    for item, prediction in results:
        try:
            prediction.validate(item.text)
        except ValueError:
            offset_valid = False
        issues = PredictionValidator(runner.components.terminology_repository).validate_prediction(
            prediction, source_text=item.text
        )
        validation_errors += len(issues)
        invalid_codes += sum(
            issue.path.endswith(".code")
            and issue.kind.startswith(("unknown_", "invalid_code"))
            for issue in issues
        )
        invalid_relations += sum(issue.kind.startswith("invalid_relation") for issue in issues)

    recall = {
        str(k): _candidate_recall(results, k)
        for k in candidate_ks
    }
    assertion_recall = _assertion_recall(results)
    output_runs = [
        [_canonical_prediction(prediction) for _, prediction in run]
        for run in repeat_results
    ]
    ordering_runs = [
        [_candidate_ordering(prediction) for _, prediction in run]
        for run in repeat_results
    ]
    return {
        "offset_validity": float(offset_valid),
        "invalid_assigned_code_rate": invalid_codes / assigned if assigned else 0.0,
        "invalid_relation_rate": invalid_relations / _relation_count(results)
        if _relation_count(results)
        else 0.0,
        "deterministic_output": len({json.dumps(run, sort_keys=True) for run in output_runs}) == 1,
        "candidate_ordering_stable": len({json.dumps(run, sort_keys=True) for run in ordering_runs}) == 1,
        "candidate_recall_at_k": recall,
        "assertion_positive_recall": assertion_recall,
        "assignment_coverage": assigned / entity_count if entity_count else 0.0,
        "validation_error_count": validation_errors,
        "entity_count": entity_count,
    }


def _candidate_recall(
    results: list[tuple[BenchmarkInput, ClinicalPrediction]], k: int
) -> float | None:
    total = hits = 0
    for item, prediction in results:
        for mention, codes in item.gold_candidates:
            if not codes:
                continue
            total += 1
            match = next((entity for entity in prediction.entities if entity.text == mention), None)
            predicted: set[str] = set()
            if match is not None:
                predicted.update(candidate.code for candidate in match.candidates[:k] if candidate.code)
                if match.code is not None:
                    predicted.add(match.code)
            hits += int(bool(predicted.intersection(codes)))
    return hits / total if total else None


def _assertion_recall(results: list[tuple[BenchmarkInput, ClinicalPrediction]]) -> float | None:
    expected = 0
    hits = 0
    for item, prediction in results:
        for mention, assertions in item.gold_assertions:
            positive = set(assertions)
            if not positive:
                continue
            expected += 1
            entity = next((candidate for candidate in prediction.entities if candidate.text == mention), None)
            if entity is not None and positive.issubset({entity.assertion.value}):
                hits += 1
    return hits / expected if expected else None


def _candidate_ordering(prediction: ClinicalPrediction) -> list[tuple[str, tuple[str | None, ...]]]:
    return [
        (entity.id, tuple(candidate.code for candidate in entity.candidates))
        for entity in prediction.entities
    ]


def _canonical_prediction(prediction: ClinicalPrediction) -> str:
    payload = prediction.to_json()
    # ``created_at`` is intentionally volatile metadata, not inference output.
    payload["metadata"] = {"pipeline_version": prediction.metadata.pipeline_version}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 6),
        "p95": round(_rank(ordered, 0.95), 6),
        "p99": round(_rank(ordered, 0.99), 6),
    }


def _rank(values: list[float], percentile: float) -> float:
    return values[min(len(values) - 1, max(0, math.ceil(len(values) * percentile) - 1))]


def _relation_count(
    results: list[tuple[BenchmarkInput, ClinicalPrediction]],
) -> int:
    return sum(len(prediction.relations) for _, prediction in results)


def _peak_rss_bytes() -> int:
    if resource is None:
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1024 if sys.platform == "darwin" else 1))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _stage_latency(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for stage in snapshot.get("stages", []):
        if stage.get("status") == "success":
            grouped.setdefault(str(stage["name"]), []).append(float(stage["elapsed_ms"]))
    return {name: _percentiles(values) for name, values in sorted(grouped.items())}


def _protected_metric(candidate: Any, baseline: Any, tolerance: float) -> bool | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) >= float(baseline) - tolerance
