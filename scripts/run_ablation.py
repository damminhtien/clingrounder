#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.evaluation.ablation import (
    AblationVariantResult,
    aggregate_traces,
    write_stage_timings_csv,
    write_summary_csv,
)
from medical_kg_nlp.evaluation.end_to_end_metrics import evaluate_predictions
from medical_kg_nlp.pipeline import PipelineOptions, PipelineRunner
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.validator import PredictionValidator
from medical_kg_nlp.utils.io import read_yaml, write_jsonl
from medical_kg_nlp.utils.run_output import create_hashed_run_dir, path_in_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured pipeline ablations with timing traces.")
    parser.add_argument("--config", default="configs/ablations.yaml", help="Ablation YAML config.")
    parser.add_argument("--output-dir", help="Override output_dir from the config.")
    parser.add_argument(
        "--run-root",
        help="Optional root for hashed run directories. Relative output paths are written under it.",
    )
    parser.add_argument("--run-label", default="ablation", help="Label embedded in the hashed run directory.")
    args = parser.parse_args()

    config = read_yaml(args.config)
    run_output = (
        create_hashed_run_dir(args.run_root, label=args.run_label, inputs=[args.config])
        if args.run_root
        else None
    )
    output_dir_override = args.output_dir
    if run_output is not None:
        output_dir_override = str(path_in_run(args.output_dir or _required_str(config, "output_dir"), run_output))

    results = run_ablation(config, output_dir_override=output_dir_override)
    summary: Any = _console_summary(results)
    if run_output is not None:
        summary = {
            "run_id": run_output.run_id,
            "run_dir": str(run_output.run_dir),
            "run_manifest": str(run_output.manifest_path),
            "results": summary,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run_ablation(
    config: dict[str, Any],
    output_dir_override: str | None = None,
) -> list[AblationVariantResult]:
    input_path = _required_str(config, "input")
    gold_path = _required_str(config, "gold")
    output_dir = Path(output_dir_override or _required_str(config, "output_dir"))
    dictionary_path = _required_str(config, "dictionary")
    abbreviation_path = _required_str(config, "abbreviations")
    variants = _variant_configs(config)

    adapter = SyntheticDatasetAdapter()
    documents = adapter.load_documents(input_path)
    documents_by_id = {document.document_id: document for document in documents}
    gold = adapter.load_gold(gold_path)

    prediction_dir = output_dir / "predictions"
    trace_dir = output_dir / "traces"
    results: list[AblationVariantResult] = []

    for variant in variants:
        name = _required_str(variant, "name")
        options_payload = variant.get("options", {})
        if not isinstance(options_payload, dict):
            raise ValueError(f"Variant {name!r} options must be a mapping.")
        options = PipelineOptions.from_mapping(cast(dict[str, object], options_payload))
        runner = PipelineRunner(
            dictionary_path=dictionary_path,
            abbreviation_path=abbreviation_path,
            pipeline_version=f"ablation:{name}",
            options=options,
        )

        start = perf_counter()
        run_results = [runner.process_document_with_trace(document) for document in documents]
        total_ms = (perf_counter() - start) * 1000

        predictions = [result.prediction for result in run_results]
        traces = [result.trace for result in run_results]
        prediction_path = prediction_dir / f"{name}.jsonl"
        trace_path = trace_dir / f"{name}.json"
        write_jsonl(prediction_path, [prediction.to_json() for prediction in predictions])
        _write_json(trace_path, [trace.to_json() for trace in traces])

        validation_issues = _count_validation_issues(runner, predictions, documents_by_id)
        metrics = evaluate_predictions(gold, predictions)
        docs_per_second = len(documents) / (total_ms / 1000) if total_ms > 0 else 0.0
        results.append(
            AblationVariantResult(
                name=name,
                metrics=metrics,
                stage_aggregates=aggregate_traces(traces),
                prediction_path=str(prediction_path),
                trace_path=str(trace_path),
                total_ms=total_ms,
                docs_per_second=docs_per_second,
                validation_issues=validation_issues,
            )
        )

    write_summary_csv(output_dir / "summary.csv", results)
    write_stage_timings_csv(output_dir / "stage_timings.csv", results)
    _write_json(output_dir / "metrics.json", [result.to_json() for result in results])
    return results


def _count_validation_issues(
    runner: PipelineRunner,
    predictions: list[ClinicalPrediction],
    documents_by_id: dict[str, Any],
) -> int:
    validator = PredictionValidator(runner.store)
    issues = 0
    for prediction in predictions:
        document = documents_by_id[prediction.document_id]
        issues += len(validator.validate_prediction(prediction, document.text))
    return issues


def _console_summary(results: list[AblationVariantResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        bottleneck = result.bottleneck_stage
        span_exact = result.metrics.get("span_exact", {})
        relation = result.metrics.get("relation", {})
        rows.append(
            {
                "variant": result.name,
                "docs_per_second": round(result.docs_per_second, 3),
                "bottleneck_stage": bottleneck.stage if bottleneck else None,
                "bottleneck_total_ms": round(bottleneck.total_ms, 3) if bottleneck else None,
                "span_exact_f1": _metric_value(span_exact, "f1"),
                "relation_f1": _metric_value(relation, "f1"),
                "validation_issues": result.validation_issues,
            }
        )
    return rows


def _metric_value(payload: Any, key: str) -> float | None:
    if isinstance(payload, dict) and isinstance(payload.get(key), (int, float)):
        return round(float(payload[key]), 6)
    return None


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string config value {key!r}.")
    return value


def _variant_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Expected non-empty variants list in ablation config.")
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("Each ablation variant must be a mapping.")
    return cast(list[dict[str, Any]], variants)


def _write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
