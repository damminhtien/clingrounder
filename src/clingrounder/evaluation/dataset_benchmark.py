"""Generic dataset benchmark runner for reproducible ClinGrounder reports.

The runner consumes a neutral JSONL contract and a pipeline profile. It does not import or know
about any competition benchmark, so a benchmark directory can be replaced without changing core
evaluation code.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping

import yaml

from clingrounder.evaluation.memory_metrics import peak_rss_bytes
from clingrounder.pipeline.config_loader import ResolvedPipelineConfig
from clingrounder.pipeline.factory import PipelineFactory
from clingrounder.schema.output import ClinicalPrediction
from clingrounder.schema.validator import PredictionValidator
from clingrounder.schema.types import CodeSystem, RelationType
from clingrounder.terminology.ports import TerminologyRepository

__all__ = [
    "compare_dataset_benchmarks",
    "run_dataset_benchmark",
    "run_dataset_benchmark_suite",
    "verify_dataset_benchmark_reference",
]

@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    """One neutral benchmark record with source-owned gold spans."""

    document_id: str
    text: str
    metadata: Mapping[str, str]
    entities: tuple[Mapping[str, Any], ...]
    relations: tuple[Mapping[str, Any], ...]


def run_dataset_benchmark(
    benchmark_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    """Run one benchmark split and write the canonical artifact directory."""

    benchmark_root = Path(benchmark_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    manifest_path = benchmark_root / "dataset_manifest.yaml"
    manifest = _load_manifest(manifest_path)
    split_payload = _mapping(manifest.get("splits"), "splits").get(split)
    if not isinstance(split_payload, Mapping):
        raise ValueError(f"Benchmark split {split!r} is not declared in {manifest_path}")
    input_path = (benchmark_root / str(split_payload["path"])).resolve()
    resolved = ResolvedPipelineConfig.load(config_path, require_profile=True)
    examples = _load_examples(
        input_path,
        entity_types=_declared_values(manifest, "entities"),
        assertions=_declared_values(manifest, "assertions"),
        code_systems=_declared_values(manifest, "code_systems"),
    )
    output_root.mkdir(parents=True, exist_ok=True)

    init_started = perf_counter()
    runtime = PipelineFactory.runtime_from_config(resolved.factory_config)
    initialization_ms = (perf_counter() - init_started) * 1000
    configuration_fingerprint = runtime.runner.components.configuration_fingerprint
    terminology_fingerprint = runtime.runner.components.terminology_fingerprint
    predictions: list[ClinicalPrediction] = []
    latencies_ms: list[float] = []
    errors: list[dict[str, Any]] = []
    validation: dict[str, Any]
    try:
        for example in examples:
            started = perf_counter()
            try:
                predictions.append(
                    runtime.runner.process_text(example.document_id, example.text, dict(example.metadata))
                )
            except Exception as error:  # noqa: BLE001 - persisted in the benchmark error artifact.
                errors.append(
                    {
                        "document_id": example.document_id,
                        "error_type": type(error).__name__,
                        "message": str(error)[:500],
                    }
                )
            finally:
                latencies_ms.append((perf_counter() - started) * 1000)
        # INVARIANT: score validity against the same terminology repository used for inference,
        # before runtime shutdown closes its resources.  A benchmark must not report a clean
        # offset/code/relation gate merely because the pipeline returned a typed object.
        validation = _validate_predictions(
            examples,
            {prediction.document_id: prediction for prediction in predictions},
            runtime.runner.components.terminology_repository,
        )
    finally:
        runtime.close()

    prediction_by_id = {prediction.document_id: prediction for prediction in predictions}
    correctness, confusion = _score(examples, prediction_by_id, validation=validation)
    git_commit = _git_commit()
    peak_rss = peak_rss_bytes()
    performance = {
        "initialization_ms": round(initialization_ms, 6),
        "documents_per_second": round(
            len(examples) / (sum(latencies_ms) / 1000), 6
        )
        if examples and sum(latencies_ms)
        else 0.0,
        "entities_per_second": round(
            correctness["entity_count"] / (sum(latencies_ms) / 1000), 6
        )
        if predictions and sum(latencies_ms)
        else 0.0,
        "document_latency_ms": _percentiles(latencies_ms),
        "peak_rss_bytes": peak_rss,
        "peak_rss_mb": round(peak_rss / (1024 * 1024), 3),
        "model_forward_pass_count": 0,
    }
    summary = {
        "schema_version": "clingrounder.benchmark-summary.v1",
        "benchmark": manifest["dataset"],
        "split": split,
        "config_fingerprint": configuration_fingerprint,
        "profile_sha256": resolved.inspection_report()["profile_sha256"],
        "terminology_fingerprint": terminology_fingerprint,
        "git_commit": git_commit,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "commit": git_commit,
        },
        "metrics": correctness,
        "performance": performance,
        "error_count": len(errors),
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "manifest.json", _artifact_manifest(
        manifest_path, input_path, config_path, summary
    ))
    _write_json(output_root / "errors.json", errors)
    _write_json(output_root / "confusion-matrices.json", confusion)
    _write_json(output_root / "runtime.json", performance)
    _write_jsonl(output_root / "predictions.jsonl", predictions)
    (output_root / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def run_dataset_benchmark_suite(
    benchmark_dir: str | Path,
    configs: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    """Run named benchmark profiles and write one deterministic ablation report.

    Each profile still gets the complete single-run artifact bundle.  The suite only adds a
    small index over those bundles, which keeps profile-specific metrics and provenance intact.
    """

    if not configs:
        raise ValueError("Benchmark suite requires at least one named config")
    normalized_configs = {
        _validate_suite_name(name): Path(path).expanduser().resolve()
        for name, path in configs.items()
    }
    if len(normalized_configs) != len(configs):
        raise ValueError("Benchmark suite config names must be unique")

    root = Path(output_dir).expanduser().resolve()
    runs: dict[str, dict[str, Any]] = {}
    for name in sorted(normalized_configs):
        summary = run_dataset_benchmark(
            benchmark_dir,
            normalized_configs[name],
            root / name,
            split=split,
        )
        runs[name] = {
            "config": str(normalized_configs[name]),
            "output": name,
            "metrics": summary["metrics"],
            "performance": summary["performance"],
            "config_fingerprint": summary["config_fingerprint"],
        }

    benchmark_root = Path(benchmark_dir).expanduser().resolve()
    manifest = _load_manifest(benchmark_root / "dataset_manifest.yaml")
    payload: dict[str, Any] = {
        "schema_version": "clingrounder.benchmark-suite.v1",
        "benchmark": manifest["dataset"],
        "split": split,
        "git_commit": _git_commit(),
        "runs": runs,
    }
    _write_json(root / "suite.json", payload)
    (root / "report.md").write_text(_render_suite_report(payload), encoding="utf-8")
    return payload


def verify_dataset_benchmark_reference(
    suite: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Verify published correctness values against a generated suite artifact.

    The reference file records one measured publication snapshot; it is not a second scorer.
    Runtime is reported but not compared because p95 latency is machine-sensitive. Correctness
    metrics, declared variants, benchmark identity, and split are checked explicitly.
    """

    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative number")
    suite_benchmark = _mapping_value(suite, "benchmark")
    reference_benchmark = _string_value(reference, "benchmark")
    suite_benchmark_id = _string_value(suite_benchmark, "id")
    if suite_benchmark_id != reference_benchmark:
        raise ValueError(
            f"Benchmark identity mismatch: suite={suite_benchmark_id!r}, "
            f"reference={reference_benchmark!r}"
        )
    suite_split = _string_value(suite, "split")
    raw_results = reference.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("reference results must be a non-empty list")
    suite_runs = _mapping_value(suite, "runs")
    variants: dict[str, dict[str, Any]] = {}
    for expected in raw_results:
        if not isinstance(expected, Mapping):
            raise ValueError("reference results must contain mappings")
        variant = _string_value(expected, "variant")
        expected_split = _string_value(expected, "split")
        actual = suite_runs.get(variant)
        if not isinstance(actual, Mapping):
            variants[variant] = {
                "correctness_match": False,
                "missing": True,
                "checks": {},
            }
            continue
        actual_metrics = _mapping_value(actual, "metrics")
        actual_performance = _mapping_value(actual, "performance")
        checks: dict[str, bool] = {}
        for name, expected_value in expected.items():
            if name in {"variant", "split", "p95_ms"}:
                continue
            checks[name] = _values_match(actual_metrics.get(name), expected_value, tolerance)
        runtime_reference = expected.get("p95_ms")
        latency = _mapping_value(actual_performance, "document_latency_ms")
        variants[variant] = {
            "correctness_match": expected_split == suite_split and all(checks.values()),
            "split_match": expected_split == suite_split,
            "missing": False,
            "checks": checks,
            "runtime": {
                "reference_p95_ms": runtime_reference,
                "measured_p95_ms": _optional_finite_number(latency.get("p95")),
                "checked": False,
            },
        }
    measurement = reference.get("measurement")
    reference_commit = (
        _optional_string(measurement.get("commit"))
        if isinstance(measurement, Mapping)
        else None
    )
    return {
        "schema_version": "clingrounder.benchmark-reference-verification.v1",
        "benchmark": reference_benchmark,
        "reference_commit": reference_commit,
        "suite_commit": _optional_string(suite.get("git_commit")),
        "runtime_checked": False,
        "verified": bool(variants)
        and all(result["correctness_match"] for result in variants.values()),
        "variants": variants,
    }


def compare_dataset_benchmarks(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a correctness-first promotion policy to two benchmark summaries.

    The comparison consumes the neutral ``summary.json`` contract, not a benchmark plugin.
    Missing metrics fail closed so a partial report cannot accidentally pass a promotion gate.
    Timing is treated as a protected metric with an explicit tolerance rather than as the
    primary objective.
    """

    if "policy" in policy and "primary" not in policy:
        policy = _mapping_value(policy, "policy")
    candidate_metrics = _mapping_value(candidate, "metrics")
    primary_policy = _mapping_value(policy, "primary")
    primary_name = _string_value(primary_policy, "metric")
    primary_minimum = _finite_number(primary_policy, "minimum_improvement")
    baseline_primary = _metric_value(baseline, primary_name)
    candidate_primary = _metric_value(candidate, primary_name)
    primary_passed = (
        baseline_primary is not None
        and candidate_primary is not None
        and candidate_primary - baseline_primary >= primary_minimum
    )

    protected: dict[str, dict[str, Any]] = {}
    protected_policy = policy.get("protected", {})
    if not isinstance(protected_policy, Mapping):
        raise ValueError("promotion policy protected must be a mapping")
    for name in sorted(protected_policy):
        raw_rule = protected_policy[name]
        if not isinstance(raw_rule, Mapping):
            raise ValueError(f"promotion policy protected.{name} must be a mapping")
        baseline_value = _metric_value(baseline, str(name))
        candidate_value = _metric_value(candidate, str(name))
        if "maximum_regression" in raw_rule:
            tolerance = _finite_number(raw_rule, "maximum_regression")
            passed = (
                baseline_value is not None
                and candidate_value is not None
                and candidate_value >= baseline_value - tolerance
            )
            rule = "maximum_regression"
        elif "maximum_regression_ratio" in raw_rule:
            tolerance = _finite_number(raw_rule, "maximum_regression_ratio")
            passed = (
                baseline_value is not None
                and candidate_value is not None
                and candidate_value <= baseline_value * (1.0 + tolerance)
            )
            rule = "maximum_regression_ratio"
        else:
            raise ValueError(
                f"promotion policy protected.{name} requires maximum_regression or "
                "maximum_regression_ratio"
            )
        protected[str(name)] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "rule": rule,
            "tolerance": tolerance,
            "passed": passed,
        }

    correctness_gates = {
        "offset_validity": candidate_metrics.get("offset_validity") == 1.0,
        "invalid_assigned_code_rate": candidate_metrics.get("invalid_assigned_code_rate") == 0.0,
        "invalid_relation_rate": candidate_metrics.get("invalid_relation_rate") == 0.0,
        "validation_error_count": candidate_metrics.get("validation_error_count") == 0,
    }
    promote = primary_passed and all(
        item["passed"] for item in protected.values()
    ) and all(correctness_gates.values())
    return {
        "schema_version": "clingrounder.dataset-promotion-comparison.v1",
        "promote": promote,
        "primary": {
            "metric": primary_name,
            "baseline": baseline_primary,
            "candidate": candidate_primary,
            "minimum_improvement": primary_minimum,
            "delta": (
                candidate_primary - baseline_primary
                if baseline_primary is not None and candidate_primary is not None
                else None
            ),
            "passed": primary_passed,
        },
        "protected": protected,
        "correctness_gates": correctness_gates,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark manifest: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "clingrounder.dataset-manifest.v1":
        raise ValueError(f"Unsupported benchmark manifest: {path}")
    if not isinstance(payload.get("dataset"), Mapping):
        raise ValueError("Benchmark manifest requires a dataset mapping")
    return payload


def _metric_value(report: Mapping[str, Any], name: str) -> float | None:
    """Read a metric from the stable summary contract, including p95 latency."""

    if name == "p95_ms":
        performance = report.get("performance")
        if isinstance(performance, Mapping):
            latency = performance.get("document_latency_ms")
            if isinstance(latency, Mapping):
                return _optional_finite_number(latency.get("p95"))
        return None
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    return _optional_finite_number(metrics.get(name))


def _values_match(actual: object, expected: object, tolerance: float) -> bool:
    """Compare reference values without accepting booleans or non-finite numbers."""

    actual_number = _optional_finite_number(actual)
    expected_number = _optional_finite_number(expected)
    if actual_number is not None or expected_number is not None:
        return (
            actual_number is not None
            and expected_number is not None
            and math.isclose(actual_number, expected_number, rel_tol=0.0, abs_tol=tolerance)
        )
    return actual == expected


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected {key} to be a mapping")
    return value


def _string_value(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected {key} to be a non-empty string")
    return value


def _finite_number(mapping: Mapping[str, Any], key: str) -> float:
    value = _optional_finite_number(mapping.get(key))
    if value is None:
        raise ValueError(f"Expected {key} to be a finite number")
    return value


def _optional_finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _declared_values(manifest: Mapping[str, Any], field: str) -> frozenset[str]:
    """Read a benchmark taxonomy from its manifest instead of a product-specific constant."""

    values = manifest.get(field)
    if not isinstance(values, list) or not values or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError(f"Benchmark manifest requires a non-empty string list: {field}")
    return frozenset(value.strip() for value in values)


def _validate_suite_name(name: str) -> str:
    """Keep suite output paths portable and prevent config-name path traversal."""

    if not name or name in {".", ".."} or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in name):
        raise ValueError(f"Invalid benchmark suite config name: {name!r}")
    return name


def _load_examples(
    path: Path,
    *,
    entity_types: frozenset[str],
    assertions: frozenset[str],
    code_systems: frozenset[str],
) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    document_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        document_id = str(row.get("document_id", "")).strip()
        text = str(row.get("text", ""))
        if not document_id or not text:
            raise ValueError(f"{path}:{line_number}: document_id and text are required")
        if document_id in document_ids:
            raise ValueError(f"{path}:{line_number}: duplicate document_id {document_id!r}")
        document_ids.add(document_id)
        raw_entities = row.get("entities")
        raw_relations = row.get("relations")
        if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
            raise ValueError(f"{path}:{line_number}: entities and relations must be arrays")
        entities = tuple(raw_entities)
        entity_ids: set[str] = set()
        for entity in entities:
            _validate_gold_entity(
                entity,
                text,
                path,
                line_number,
                entity_types=entity_types,
                assertions=assertions,
                code_systems=code_systems,
            )
            entity_id = str(entity["id"]).strip()
            if entity_id in entity_ids:
                raise ValueError(f"{path}:{line_number}: duplicate entity id {entity_id!r}")
            entity_ids.add(entity_id)
        relation_ids: set[str] = set()
        for relation in raw_relations:
            _validate_gold_relation(relation, entity_ids, relation_ids, path, line_number)
        examples.append(
            BenchmarkExample(
                document_id=document_id,
                text=text,
                metadata={str(k): str(v) for k, v in row.get("metadata", {}).items()},
                entities=entities,
                relations=tuple(raw_relations),
            )
        )
    if not examples:
        raise ValueError(f"Benchmark input {path} contains no documents")
    return examples


def _validate_gold_entity(
    entity: object,
    text: str,
    path: Path,
    line_number: int,
    *,
    entity_types: frozenset[str],
    assertions: frozenset[str],
    code_systems: frozenset[str],
) -> None:
    if not isinstance(entity, Mapping):
        raise ValueError(f"{path}:{line_number}: entity must be an object")
    entity_id = str(entity.get("id", "")).strip()
    if not entity_id:
        raise ValueError(f"{path}:{line_number}: entity id is required")
    entity_type = entity.get("type")
    assertion = entity.get("assertion")
    code_system = entity.get("code_system")
    code = entity.get("code")
    if entity_type not in entity_types:
        raise ValueError(f"{path}:{line_number}: unsupported entity type {entity_type!r}")
    if assertion not in assertions:
        raise ValueError(f"{path}:{line_number}: unsupported assertion {assertion!r}")
    if code_system not in code_systems:
        raise ValueError(f"{path}:{line_number}: unsupported code system {code_system!r}")
    if code is not None and (not isinstance(code, str) or not code.strip()):
        raise ValueError(f"{path}:{line_number}: code must be null or a non-empty string")
    if code_system == CodeSystem.NONE.value and code is not None:
        raise ValueError(f"{path}:{line_number}: NONE code system requires null code")
    if code_system != CodeSystem.NONE.value and code is None:
        raise ValueError(f"{path}:{line_number}: assigned code system requires a code")
    span = entity.get("span")
    if not isinstance(span, list | tuple) or len(span) != 2:
        raise ValueError(f"{path}:{line_number}: entity span must contain two offsets")
    start, end = span
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
        raise ValueError(f"{path}:{line_number}: invalid entity span {span!r}")
    if text[start:end] != entity.get("text"):
        raise ValueError(f"{path}:{line_number}: entity span/text mismatch")


def _validate_gold_relation(
    relation: object,
    entity_ids: set[str],
    relation_ids: set[str],
    path: Path,
    line_number: int,
) -> None:
    """Validate neutral relation endpoints before they can affect benchmark metrics."""

    if not isinstance(relation, Mapping):
        raise ValueError(f"{path}:{line_number}: relation must be an object")
    relation_id = str(relation.get("id", "")).strip()
    head = str(relation.get("head", "")).strip()
    tail = str(relation.get("tail", "")).strip()
    relation_type = str(relation.get("type", "")).strip()
    if not relation_id or not head or not tail or not relation_type:
        raise ValueError(f"{path}:{line_number}: relation id/head/tail/type are required")
    if relation_id in relation_ids:
        raise ValueError(f"{path}:{line_number}: duplicate relation id {relation_id!r}")
    relation_ids.add(relation_id)
    if head == tail:
        raise ValueError(f"{path}:{line_number}: relation {relation_id!r} cannot self-loop")
    if head not in entity_ids or tail not in entity_ids:
        raise ValueError(
            f"{path}:{line_number}: relation {relation_id!r} references an unknown entity"
        )
    try:
        relation_enum = RelationType(relation_type)
    except ValueError as error:
        raise ValueError(
            f"{path}:{line_number}: unsupported relation type {relation_type!r}"
        ) from error
    if relation_enum is RelationType.UNKNOWN:
        raise ValueError(f"{path}:{line_number}: UNKNOWN relation type is not valid gold")


def _score(
    examples: list[BenchmarkExample],
    predictions: Mapping[str, ClinicalPrediction],
    *,
    validation: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gold_total = pred_total = true_positive = 0
    gold_by_type: Counter[str] = Counter()
    pred_by_type: Counter[str] = Counter()
    tp_by_type: Counter[str] = Counter()
    assertion_counts: Counter[tuple[str, str]] = Counter()
    linking_total = 0
    assigned_prediction_count = 0
    linking_hits: Counter[int] = Counter()
    top1_hits = 0
    gold_relation_keys: set[tuple[str, str, str, str]] = set()
    pred_relation_keys: set[tuple[str, str, str, str]] = set()
    errors_by_document: list[dict[str, Any]] = []

    for example in examples:
        prediction = predictions.get(example.document_id)
        predicted_entities = [] if prediction is None else prediction.entities
        gold_keys = {(tuple(item["span"]), str(item["type"])) for item in example.entities}
        pred_keys = {(entity.span, entity.type.value) for entity in predicted_entities}
        true_positive += len(gold_keys & pred_keys)
        gold_total += len(gold_keys)
        pred_total += len(pred_keys)
        for item in example.entities:
            gold_by_type[str(item["type"])] += 1
            key = (tuple(item["span"]), str(item["type"]))
            matched = next((entity for entity in predicted_entities if (entity.span, entity.type.value) == key), None)
            expected_code = item.get("code")
            # INVARIANT: linking recall counts every coded gold mention, including a mention
            # missed by NER. Otherwise extraction failures disappear from linking metrics.
            if expected_code:
                linking_total += 1
            if matched is not None:
                tp_by_type[str(item["type"])] += 1
                assertion_counts[(str(item["assertion"]), matched.assertion.value)] += 1
                if expected_code:
                    codes = [candidate.code for candidate in matched.candidates]
                    if matched.code:
                        codes.insert(0, matched.code)
                        assigned_prediction_count += 1
                    for k in (1, 5, 10):
                        if expected_code in codes[:k]:
                            linking_hits[k] += 1
                    top1_hits += int(bool(codes) and codes[0] == expected_code)
        for entity in predicted_entities:
            pred_by_type[entity.type.value] += 1
        if prediction is not None:
            # Relation IDs in neutral gold are independent from pipeline IDs. Map gold entity
            # IDs through exact span/type matches before comparing relation endpoints.
            gold_to_prediction_id = {
                str(item["id"]): matched.id
                for item in example.entities
                if (matched := next(
                    (
                        entity
                        for entity in predicted_entities
                        if (entity.span, entity.type.value)
                        == (tuple(item["span"]), str(item["type"]))
                    ),
                    None,
                ))
                is not None
            }
            gold_relation_keys.update(
                (
                    example.document_id,
                    gold_to_prediction_id.get(str(item["head"]), str(item["head"])),
                    gold_to_prediction_id.get(str(item["tail"]), str(item["tail"])),
                    str(item["type"]),
                )
                for item in example.relations
            )
            pred_relation_keys.update(
                (example.document_id, relation.head, relation.tail, relation.type.value)
                for relation in prediction.relations
            )
        if len(gold_keys - pred_keys) or len(pred_keys - gold_keys):
            errors_by_document.append(
                {
                    "document_id": example.document_id,
                    "missing": sorted(gold_keys - pred_keys),
                    "spurious": sorted(pred_keys - gold_keys),
                }
            )

    precision = _ratio(true_positive, pred_total)
    recall = _ratio(true_positive, gold_total)
    overlap_true_positive = _overlap_matches(examples, predictions)
    overlap_precision = _ratio(overlap_true_positive, pred_total)
    overlap_recall = _ratio(overlap_true_positive, gold_total)
    type_rows = {}
    for entity_type in sorted(set(gold_by_type) | set(pred_by_type)):
        type_precision = _ratio(tp_by_type[entity_type], pred_by_type[entity_type])
        type_recall = _ratio(tp_by_type[entity_type], gold_by_type[entity_type])
        type_rows[entity_type] = {
            "precision": type_precision,
            "recall": type_recall,
            "f1": _f1(type_precision, type_recall),
            "gold": gold_by_type[entity_type],
            "predicted": pred_by_type[entity_type],
        }
    assertion_labels = sorted({key[0] for key in assertion_counts})
    assertion_f1 = _assertion_macro_f1(assertion_counts, assertion_labels)
    positive_labels = [label for label in assertion_labels if label not in {"PRESENT", "UNKNOWN"}]
    positive_assertion_f1 = _assertion_macro_f1(assertion_counts, positive_labels)
    assertion_accuracy = _ratio(
        sum(count for (gold, predicted), count in assertion_counts.items() if gold == predicted),
        sum(assertion_counts.values()),
    )
    relation_tp = len(gold_relation_keys & pred_relation_keys)
    relation_precision = _ratio(relation_tp, len(pred_relation_keys))
    relation_recall = _ratio(relation_tp, len(gold_relation_keys))
    reciprocal_rank_total = 0.0
    for example in examples:
        prediction = predictions.get(example.document_id)
        if prediction is None:
            continue
        for item in example.entities:
            expected_code = item.get("code")
            if not expected_code:
                continue
            matched = next(
                (
                    entity
                    for entity in prediction.entities
                    if (entity.span, entity.type.value)
                    == (tuple(item["span"]), str(item["type"]))
                ),
                None,
            )
            if matched is None:
                continue
            codes = [candidate.code for candidate in matched.candidates]
            if matched.code:
                codes.insert(0, matched.code)
            try:
                reciprocal_rank_total += 1.0 / (codes.index(expected_code) + 1)
            except ValueError:
                pass

    metrics = {
        "entity_exact_micro_f1": _f1(precision, recall),
        "entity_exact_precision": precision,
        "entity_exact_recall": recall,
        "entity_overlap_micro_f1": _f1(overlap_precision, overlap_recall),
        "entity_count": pred_total,
        "entity_by_type": type_rows,
        "assertion_accuracy": assertion_accuracy,
        "assertion_macro_f1": assertion_f1,
        "assertion_positive_macro_f1": positive_assertion_f1,
        "linking_recall_at_1": _ratio(linking_hits[1], linking_total),
        "linking_recall_at_5": _ratio(linking_hits[5], linking_total),
        "linking_recall_at_10": _ratio(linking_hits[10], linking_total),
        "linking_top1_accuracy": _ratio(top1_hits, linking_total),
        "linking_mrr": _ratio(reciprocal_rank_total, linking_total),
        "assigned_prediction_count": assigned_prediction_count,
        "relation_micro_f1": _f1(relation_precision, relation_recall),
        "relation_gold_count": len(gold_relation_keys),
        "relation_predicted_count": len(pred_relation_keys),
        "linkable_gold_count": linking_total,
        "assignment_coverage": _ratio(assigned_prediction_count, pred_total),
        "offset_validity": 1.0,
        "validation_error_count": 0,
        "document_error_count": len(errors_by_document),
    }
    if validation is not None:
        metrics.update(validation)
    return (
        metrics,
        {
            "assertion": {
                "labels": assertion_labels,
                "counts": {f"{gold}->{pred}": count for (gold, pred), count in sorted(assertion_counts.items())},
            },
            "documents": errors_by_document,
        },
    )


def _validate_predictions(
    examples: list[BenchmarkExample],
    predictions: Mapping[str, ClinicalPrediction],
    terminology: TerminologyRepository | None,
) -> dict[str, Any]:
    """Compute fail-closed correctness gates from actual benchmark predictions.

    The dataset scorer intentionally remains independent of any benchmark-specific gold labels.
    This pass checks only output invariants and active terminology membership, so a malformed
    adapter cannot make a benchmark look successful by bypassing release validation.
    """

    validator = PredictionValidator(terminology)
    assigned_count = 0
    relation_count = 0
    validation_error_count = 0
    error_kinds: Counter[str] = Counter()
    invalid_assigned_entities: set[tuple[str, int]] = set()
    invalid_relations: set[tuple[str, int]] = set()
    offset_valid = len(predictions) == len(examples)

    for example in examples:
        prediction = predictions.get(example.document_id)
        if prediction is None:
            continue
        assigned_entity_indices = {
            index for index, entity in enumerate(prediction.entities) if entity.code is not None
        }
        assigned_count += len(assigned_entity_indices)
        relation_count += len(prediction.relations)
        issues = validator.validate_prediction(prediction, source_text=example.text)
        validation_error_count += len(issues)
        error_kinds.update(issue.kind for issue in issues)
        offset_valid = offset_valid and not any(issue.kind == "offset" for issue in issues)
        for issue in issues:
            entity_index = _indexed_path(issue.path, "$.entities[")
            if (
                entity_index is not None
                and entity_index in assigned_entity_indices
                and ".candidates[" not in issue.path
                and issue.kind.startswith(("unknown_", "invalid_code"))
            ):
                invalid_assigned_entities.add((example.document_id, entity_index))
            relation_index = _indexed_path(issue.path, "$.relations[")
            if relation_index is not None:
                invalid_relations.add((example.document_id, relation_index))

    return {
        "offset_validity": float(offset_valid),
        "invalid_assigned_code_rate": (
            len(invalid_assigned_entities) / assigned_count if assigned_count else 0.0
        ),
        "invalid_relation_rate": (
            len(invalid_relations) / relation_count if relation_count else 0.0
        ),
        "validation_error_count": validation_error_count,
        "validation_error_kinds": dict(sorted(error_kinds.items())),
        "missing_prediction_count": len(examples) - len(predictions),
    }


def _indexed_path(path: str, prefix: str) -> int | None:
    """Extract a list index from a validator path without parsing arbitrary user input."""

    if not path.startswith(prefix):
        return None
    index_text, separator, _ = path[len(prefix) :].partition("]")
    if not separator or not index_text.isdigit():
        return None
    return int(index_text)


def _artifact_manifest(
    manifest_path: Path,
    input_path: Path,
    config_path: str | Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "clingrounder.benchmark-artifact.v1",
        "benchmark_manifest_sha256": _sha256_file(manifest_path),
        "input_sha256": _sha256_file(input_path),
        "config_sha256": _sha256_file(Path(config_path).resolve()),
        "git_commit": summary.get("environment", {}).get("commit"),
        "benchmark": summary.get("benchmark"),
        "split": summary.get("split"),
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    performance = summary["performance"]
    return "\n".join(
        [
            f"# {summary['benchmark']['id']}",
            "",
            f"Split: `{summary['split']}`",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Entity exact micro-F1 | {metrics['entity_exact_micro_f1']:.4f} |",
            f"| Entity overlap micro-F1 | {metrics['entity_overlap_micro_f1']:.4f} |",
            f"| Assertion accuracy | {metrics['assertion_accuracy']:.4f} |",
            f"| Assertion macro-F1 | {metrics['assertion_macro_f1']:.4f} |",
            f"| Linking Recall@5 | {metrics['linking_recall_at_5']:.4f} |",
            f"| Linking MRR | {metrics['linking_mrr']:.4f} |",
            f"| Linking Top-1 | {metrics['linking_top1_accuracy']:.4f} |",
            f"| Relation micro-F1 | {metrics['relation_micro_f1']:.4f} |",
            "",
            "## Runtime",
            "",
            f"- Initialization: `{performance['initialization_ms']:.2f} ms`",
            f"- Documents/second: `{performance['documents_per_second']:.4f}`",
            f"- p95 document latency: `{performance['document_latency_ms']['p95']:.2f} ms`",
            "",
            "Synthetic pilot results are not clinical validation evidence.",
            "",
        ]
    )


def _render_suite_report(payload: Mapping[str, Any]) -> str:
    """Render a compact ablation table without hiding per-run provenance."""

    lines = [
        f"# {payload['benchmark']['id']} suite",
        "",
        f"Split: `{payload['split']}`",
        "",
        "| Variant | Entity exact F1 | Assertion macro-F1 | Recall@5 | Top-1 | Relation F1 | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, run in sorted(payload["runs"].items()):
        metrics = run["metrics"]
        performance = run["performance"]
        relation = metrics["relation_micro_f1"]
        relation_value = "N/A" if metrics["relation_gold_count"] == 0 else f"{relation:.4f}"
        lines.append(
            "| {name} | {entity:.4f} | {assertion:.4f} | {recall:.4f} | {top1:.4f} | "
            "{relation} | {p95:.2f} |".format(
                name=name,
                entity=metrics["entity_exact_micro_f1"],
                assertion=metrics["assertion_macro_f1"],
                recall=metrics["linking_recall_at_5"],
                top1=metrics["linking_top1_accuracy"],
                relation=relation_value,
                p95=performance["document_latency_ms"]["p95"],
            )
        )
    lines.extend(
        [
            "",
            "Each row is backed by the complete artifact bundle in the matching subdirectory.",
            "Runtime values are machine-dependent; correctness values on a synthetic pilot are "
            "not clinical validation evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, predictions: list[ClinicalPrediction]) -> None:
    rows: list[str] = []
    for prediction in predictions:
        payload = prediction.to_json()
        # INVARIANT: benchmark artifacts must be byte-stable; wall-clock creation time belongs
        # in runtime traces, not in canonical prediction records.
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("created_at", None)
        rows.append(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    path.write_text("".join(rows), encoding="utf-8")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _overlap_matches(
    examples: list[BenchmarkExample],
    predictions: Mapping[str, ClinicalPrediction],
) -> int:
    """Count deterministic one-to-one same-type span overlaps for diagnostic reporting."""

    matched_total = 0
    for example in examples:
        prediction = predictions.get(example.document_id)
        if prediction is None:
            continue
        available = set(range(len(prediction.entities)))
        for gold in example.entities:
            gold_start, gold_end = gold["span"]
            candidates = []
            for index in available:
                entity = prediction.entities[index]
                if entity.type.value != str(gold["type"]):
                    continue
                overlap = min(gold_end, entity.span[1]) - max(gold_start, entity.span[0])
                if overlap > 0:
                    candidates.append(
                        (
                            -overlap,
                            abs((gold_end - gold_start) - (entity.span[1] - entity.span[0])),
                            entity.span,
                            index,
                        )
                    )
            if candidates:
                _, _, _, selected = min(candidates)
                available.remove(selected)
                matched_total += 1
    return matched_total


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _assertion_macro_f1(counts: Counter[tuple[str, str]], labels: list[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = counts[(label, label)]
        fp = sum(counts[(other, label)] for other in labels if other != label)
        fn = sum(counts[(label, other)] for other in labels if other != label)
        scores.append(_f1(_ratio(tp, tp + fp), _ratio(tp, tp + fn)))
    return sum(scores) / len(scores) if scores else 0.0


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 6),
        "p95": round(ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.95)))], 6),
        "p99": round(ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.99)))], 6),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
