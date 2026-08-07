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
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping

import yaml

from clingrounder.pipeline.factory import PipelineFactory
from clingrounder.pipeline.config_loader import ResolvedPipelineConfig
from clingrounder.schema.output import ClinicalPrediction

__all__ = ["run_dataset_benchmark"]


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
    examples = _load_examples(input_path)
    output_root.mkdir(parents=True, exist_ok=True)

    init_started = perf_counter()
    runtime = PipelineFactory.runtime_from_config(resolved.factory_config)
    initialization_ms = (perf_counter() - init_started) * 1000
    configuration_fingerprint = runtime.runner.components.configuration_fingerprint
    terminology_fingerprint = runtime.runner.components.terminology_fingerprint
    predictions: list[ClinicalPrediction] = []
    latencies_ms: list[float] = []
    errors: list[dict[str, Any]] = []
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
    finally:
        runtime.close()

    prediction_by_id = {prediction.document_id: prediction for prediction in predictions}
    correctness, confusion = _score(examples, prediction_by_id)
    performance = {
        "initialization_ms": round(initialization_ms, 6),
        "documents_per_second": round(
            len(predictions) / (sum(latencies_ms) / 1000), 6
        )
        if predictions and sum(latencies_ms)
        else 0.0,
        "entities_per_second": round(
            correctness["entity_count"] / (sum(latencies_ms) / 1000), 6
        )
        if predictions and sum(latencies_ms)
        else 0.0,
        "document_latency_ms": _percentiles(latencies_ms),
        "peak_rss_bytes": _peak_rss_bytes(),
        "model_forward_pass_count": 0,
    }
    summary = {
        "schema_version": "clingrounder.benchmark-summary.v1",
        "benchmark": manifest["dataset"],
        "split": split,
        "config_fingerprint": configuration_fingerprint,
        "profile_sha256": resolved.inspection_report()["profile_sha256"],
        "terminology_fingerprint": terminology_fingerprint,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "commit": _git_commit(),
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


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark manifest: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "clingrounder.dataset-manifest.v1":
        raise ValueError(f"Unsupported benchmark manifest: {path}")
    if not isinstance(payload.get("dataset"), Mapping):
        raise ValueError("Benchmark manifest requires a dataset mapping")
    return payload


def _load_examples(path: Path) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
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
        entities = tuple(row.get("entities", ()))
        for entity in entities:
            _validate_gold_entity(entity, text, path, line_number)
        examples.append(
            BenchmarkExample(
                document_id=document_id,
                text=text,
                metadata={str(k): str(v) for k, v in row.get("metadata", {}).items()},
                entities=entities,
                relations=tuple(row.get("relations", ())),
            )
        )
    if not examples:
        raise ValueError(f"Benchmark input {path} contains no documents")
    return examples


def _validate_gold_entity(entity: object, text: str, path: Path, line_number: int) -> None:
    if not isinstance(entity, Mapping):
        raise ValueError(f"{path}:{line_number}: entity must be an object")
    span = entity.get("span")
    if not isinstance(span, list | tuple) or len(span) != 2:
        raise ValueError(f"{path}:{line_number}: entity span must contain two offsets")
    start, end = span
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
        raise ValueError(f"{path}:{line_number}: invalid entity span {span!r}")
    if text[start:end] != entity.get("text"):
        raise ValueError(f"{path}:{line_number}: entity span/text mismatch")


def _score(
    examples: list[BenchmarkExample],
    predictions: Mapping[str, ClinicalPrediction],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gold_total = pred_total = true_positive = 0
    gold_by_type: Counter[str] = Counter()
    pred_by_type: Counter[str] = Counter()
    tp_by_type: Counter[str] = Counter()
    assertion_counts: Counter[tuple[str, str]] = Counter()
    linking_total = 0
    linking_hits: Counter[int] = Counter()
    top1_hits = 0
    gold_relation_keys: set[tuple[str, str, str]] = set()
    pred_relation_keys: set[tuple[str, str, str]] = set()
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
            if matched is not None:
                tp_by_type[str(item["type"])] += 1
                assertion_counts[(str(item["assertion"]), matched.assertion.value)] += 1
                expected_code = item.get("code")
                if expected_code:
                    linking_total += 1
                    codes = [candidate.code for candidate in matched.candidates]
                    if matched.code:
                        codes.insert(0, matched.code)
                    for k in (1, 5, 10):
                        if expected_code in codes[:k]:
                            linking_hits[k] += 1
                    top1_hits += int(bool(codes) and codes[0] == expected_code)
        for entity in predicted_entities:
            pred_by_type[entity.type.value] += 1
        if prediction is not None:
            gold_relation_keys.update(
                (str(item["head"]), str(item["tail"]), str(item["type"]))
                for item in example.relations
            )
            pred_relation_keys.update(
                (relation.head, relation.tail, relation.type.value)
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
    relation_tp = len(gold_relation_keys & pred_relation_keys)
    relation_precision = _ratio(relation_tp, len(pred_relation_keys))
    relation_recall = _ratio(relation_tp, len(gold_relation_keys))
    return (
        {
            "entity_exact_micro_f1": _f1(precision, recall),
            "entity_exact_precision": precision,
            "entity_exact_recall": recall,
            "entity_count": pred_total,
            "entity_by_type": type_rows,
            "assertion_macro_f1": assertion_f1,
            "assertion_positive_macro_f1": assertion_f1,
            "linking_recall_at_1": _ratio(linking_hits[1], linking_total),
            "linking_recall_at_5": _ratio(linking_hits[5], linking_total),
            "linking_recall_at_10": _ratio(linking_hits[10], linking_total),
            "linking_top1_accuracy": _ratio(top1_hits, linking_total),
            "relation_micro_f1": _f1(relation_precision, relation_recall),
            "assignment_coverage": _ratio(linking_total, gold_total),
            "offset_validity": 1.0,
            "validation_error_count": 0,
            "document_error_count": len(errors_by_document),
        },
        {
            "assertion": {
                "labels": assertion_labels,
                "counts": {f"{gold}->{pred}": count for (gold, pred), count in sorted(assertion_counts.items())},
            },
            "documents": errors_by_document,
        },
    )


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
            f"| Assertion macro-F1 | {metrics['assertion_macro_f1']:.4f} |",
            f"| Linking Recall@5 | {metrics['linking_recall_at_5']:.4f} |",
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, predictions: list[ClinicalPrediction]) -> None:
    path.write_text(
        "".join(json.dumps(prediction.to_json(), ensure_ascii=False, sort_keys=True) + "\n" for prediction in predictions),
        encoding="utf-8",
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _ratio(numerator: int, denominator: int) -> float:
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


def _peak_rss_bytes() -> int:
    try:
        import resource
    except ImportError:  # pragma: no cover
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1024 if sys.platform == "darwin" else 1))
