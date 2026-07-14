#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.evaluation.phase1 import (
    Phase1ExportPolicy,
    Phase1SelectiveExportConfig,
    load_calibrated_assertion_map,
    load_reviewed_candidate_map,
    load_phase1_text_documents,
    validate_phase1_submission_dir,
    validate_phase1_submission_zip,
    write_phase1_output_dir,
    zip_phase1_output_dir,
)
from medical_kg_nlp.pipeline.parallel_batch import (
    ParallelBackend,
    ParallelBatchOptions,
    run_batch_with_trace_parallel,
)
from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.utils.io import read_yaml, write_jsonl
from medical_kg_nlp.utils.run_output import create_hashed_run_dir, path_in_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase 1 flat JSON files and optional output.zip from input TXT files.",
    )
    parser.add_argument(
        "--config",
        default="configs/phase1_submission.yaml",
        help="Phase 1 submission YAML config. CLI values override this file.",
    )
    parser.add_argument("--input-dir", help="Directory containing 1.txt..100.txt.")
    parser.add_argument("--output-dir", help="Directory that will contain 1.json..100.json.")
    parser.add_argument("--zip", dest="zip_path", help="Optional submission zip path.")
    parser.add_argument(
        "--run-root",
        help="Optional root for hashed run directories. Relative output paths are written under it.",
    )
    parser.add_argument("--run-label", help="Label embedded in the hashed run directory.")
    parser.add_argument(
        "--mode",
        choices=("entity_only", "selective", "full", "custom"),
        help="Execution contract. Config defaults to custom when omitted.",
    )
    parser.add_argument(
        "--pred", help="Optional internal prediction JSONL to export instead of running."
    )
    parser.add_argument(
        "--internal-predictions",
        help="Optional path for the full internal prediction JSONL with decision provenance.",
    )
    parser.add_argument(
        "--traces",
        help="Optional path for per-document pipeline trace JSONL.",
    )
    parser.add_argument(
        "--runtime-metrics",
        help="Optional JSON path for batch initialization and processing timing.",
    )
    parser.add_argument(
        "--dictionary",
        help="Dictionary JSONL used for pipeline linking and candidate validation.",
    )
    parser.add_argument(
        "--abbreviations",
        help="Abbreviation JSONL used when running the pipeline.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Max candidates exported per codable Phase 1 entity. Default keeps candidate sets precise.",
    )
    parser.add_argument(
        "--assertion-policy",
        choices=("empty", "pipeline", "selective"),
        help="Export assertions from the pipeline or abstain with empty lists.",
    )
    parser.add_argument(
        "--candidate-policy",
        choices=("empty", "pipeline", "selective"),
        help="Export candidates from the pipeline or abstain with empty lists.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Expected JSON files in the official ZIP structure.",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=("serial", "thread", "process"),
        help="Pipeline execution backend when --pred is omitted.",
    )
    parser.add_argument("--workers", type=int, help="Number of document workers.")
    parser.add_argument("--chunksize", type=int, help="Document chunksize for process workers.")
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Collect document worker errors before raising a batch error.",
    )
    parser.add_argument(
        "--no-strict-validation",
        action="store_true",
        help="Write artifacts even if Phase 1 validation issues are found.",
    )
    args = parser.parse_args()
    config = read_yaml(args.config) if args.config else {}
    parallel_config = _mapping(config.get("parallel"), "parallel")
    pipeline_config = _mapping(config.get("pipeline"), "pipeline")

    input_dir = _required_str(args.input_dir or config.get("input_dir"), "input_dir", parser)
    output_dir_arg = _required_str(
        args.output_dir or config.get("output_dir"), "output_dir", parser
    )
    zip_arg = _optional_str(args.zip_path if args.zip_path is not None else config.get("zip"))
    run_root = _optional_str(args.run_root if args.run_root is not None else config.get("run_root"))
    run_label = _required_str(
        args.run_label or config.get("run_label") or "phase1", "run_label", parser
    )
    mode = str(args.mode or config.get("mode") or "custom").strip().lower()
    pred_path = _optional_str(args.pred if args.pred is not None else config.get("pred"))
    internal_predictions_arg = _optional_str(
        args.internal_predictions
        if args.internal_predictions is not None
        else config.get("internal_predictions")
    )
    traces_arg = _optional_str(
        args.traces if args.traces is not None else config.get("traces")
    )
    runtime_metrics_arg = _optional_str(
        args.runtime_metrics
        if args.runtime_metrics is not None
        else config.get("runtime_metrics")
    )
    dictionary_path = _required_str(
        args.dictionary or config.get("dictionary") or "data/dictionaries/seed_concepts.jsonl",
        "dictionary",
        parser,
    )
    abbreviation_path = _required_str(
        args.abbreviations
        or config.get("abbreviations")
        or "data/dictionaries/abbreviations.jsonl",
        "abbreviations",
        parser,
    )
    normalization_dictionary_path = _optional_str(
        config.get("normalization_dictionary")
    )
    recognition_dictionary_path = _optional_str(config.get("recognition_dictionary"))
    max_candidates = _int_setting(
        args.max_candidates, config.get("max_candidates"), 5, "max_candidates"
    )
    assertion_policy = _export_policy(
        args.assertion_policy or config.get("assertion_policy"), "assertion_policy"
    )
    candidate_policy = _export_policy(
        args.candidate_policy or config.get("candidate_policy"), "candidate_policy"
    )
    expected_count = _int_setting(
        args.expected_count, config.get("expected_count"), 100, "expected_count"
    )
    parallel_backend = str(args.parallel_backend or parallel_config.get("backend") or "process")
    if parallel_backend not in {"serial", "thread", "process"}:
        raise ValueError("parallel.backend must be one of: serial, thread, process.")
    workers = _int_setting(args.workers, parallel_config.get("workers"), 1, "parallel.workers")
    chunksize = _int_setting(
        args.chunksize, parallel_config.get("chunksize"), 4, "parallel.chunksize"
    )
    fail_fast = (
        False
        if args.no_fail_fast
        else _bool_setting(parallel_config.get("fail_fast"), True, "parallel.fail_fast")
    )
    strict_validation = (
        False
        if args.no_strict_validation
        else _bool_setting(config.get("strict_validation"), True, "strict_validation")
    )
    pipeline_options = PipelineOptions.from_mapping(pipeline_config)
    selective_config = _selective_config(config, assertion_policy, candidate_policy)
    _validate_submission_mode(
        mode,
        assertion_policy,
        candidate_policy,
        pipeline_options,
        selective_config,
    )

    run_output = (
        create_hashed_run_dir(
            run_root,
            label=run_label,
            resolved_config=config,
            inputs=[
                input_dir,
                dictionary_path,
                recognition_dictionary_path or "",
                normalization_dictionary_path or dictionary_path,
                pred_path or "pipeline",
                f"assertions={assertion_policy}",
                f"candidates={candidate_policy}",
                f"mode={mode}",
            ],
        )
        if run_root
        else None
    )
    documents = load_phase1_text_documents(input_dir)
    batch_runtime: dict[str, object] = {}
    if pred_path:
        predictions = SyntheticDatasetAdapter().load_gold(pred_path)
        traces = []
        batch_runtime = {
            "backend": "precomputed",
            "document_count": len(documents),
            "worker_count": 0,
        }
    else:
        run_results = run_batch_with_trace_parallel(
            documents,
            dictionary_path=dictionary_path,
            abbreviation_path=abbreviation_path,
            recognition_dictionary_path=recognition_dictionary_path,
            normalization_dictionary_path=normalization_dictionary_path,
            pipeline_options=pipeline_options,
            parallel_options=ParallelBatchOptions(
                backend=cast(ParallelBackend, parallel_backend),
                max_workers=workers,
                chunksize=chunksize,
                fail_fast=fail_fast,
            ),
            runtime_metrics=batch_runtime,
        )
        predictions = [result.prediction for result in run_results]
        traces = [result.trace.to_json() for result in run_results]

    internal_predictions_path = (
        path_in_run(internal_predictions_arg, run_output)
        if internal_predictions_arg
        else None
    )
    if internal_predictions_path is not None:
        write_jsonl(
            internal_predictions_path,
            [prediction.to_json() for prediction in predictions],
        )
    traces_path = path_in_run(traces_arg, run_output) if traces_arg else None
    if traces_path is not None:
        write_jsonl(traces_path, traces)
    runtime_metrics_path = (
        path_in_run(runtime_metrics_arg, run_output) if runtime_metrics_arg else None
    )
    if runtime_metrics_path is not None:
        _write_json(runtime_metrics_path, batch_runtime)

    output_dir = path_in_run(output_dir_arg, run_output)
    source_text_by_document = {document.document_id: document.text for document in documents}
    write_phase1_output_dir(
        predictions,
        output_dir,
        max_candidates=max_candidates,
        source_text_by_document=source_text_by_document,
        assertion_policy=assertion_policy,
        candidate_policy=candidate_policy,
        selective_config=selective_config,
    )
    primary_dictionary = DictionaryStore.from_jsonl(dictionary_path)
    if normalization_dictionary_path:
        normalization_dictionary = DictionaryStore.from_jsonl(
            normalization_dictionary_path
        )
        dictionary = DictionaryStore(
            list(
                {
                    entry.concept_id: entry
                    for entry in (
                        *primary_dictionary.entries,
                        *normalization_dictionary.entries,
                    )
                }.values()
            )
        )
    else:
        dictionary = primary_dictionary
    issues = [
        issue.to_json()
        for issue in validate_phase1_submission_dir(
            input_dir,
            output_dir,
            dictionary=dictionary,
        )
    ]
    zip_path = path_in_run(zip_arg, run_output) if zip_arg else None
    summary = _summary(
        run_output=run_output,
        documents=len(documents),
        predictions=len(predictions),
        output_dir=output_dir,
        zip_path=zip_path,
        traces=len(traces),
        issues=issues,
        strict_validation=strict_validation,
        pipeline_options=pipeline_options,
        assertion_policy=assertion_policy,
        candidate_policy=candidate_policy,
        mode=mode,
        internal_predictions_path=internal_predictions_path,
        traces_path=traces_path,
        runtime_metrics_path=runtime_metrics_path,
        batch_runtime=batch_runtime,
    )
    if issues and strict_validation:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(1)

    if zip_arg:
        zip_phase1_output_dir(output_dir, zip_path or zip_arg)
        issues.extend(
            issue.to_json()
            for issue in validate_phase1_submission_zip(
                zip_path or zip_arg,
                input_dir=input_dir,
                dictionary=dictionary,
                expected_count=expected_count,
            )
        )

    summary = _summary(
        run_output=run_output,
        documents=len(documents),
        predictions=len(predictions),
        output_dir=output_dir,
        zip_path=zip_path,
        traces=len(traces),
        issues=issues,
        strict_validation=strict_validation,
        pipeline_options=pipeline_options,
        assertion_policy=assertion_policy,
        candidate_policy=candidate_policy,
        mode=mode,
        internal_predictions_path=internal_predictions_path,
        traces_path=traces_path,
        runtime_metrics_path=runtime_metrics_path,
        batch_runtime=batch_runtime,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if issues and strict_validation:
        raise SystemExit(1)


def _summary(
    *,
    run_output: Any,
    documents: int,
    predictions: int,
    output_dir: Path,
    zip_path: Path | None,
    traces: int,
    issues: list[dict[str, str]],
    strict_validation: bool,
    pipeline_options: PipelineOptions,
    assertion_policy: Phase1ExportPolicy,
    candidate_policy: Phase1ExportPolicy,
    mode: str,
    internal_predictions_path: Path | None,
    traces_path: Path | None,
    runtime_metrics_path: Path | None,
    batch_runtime: dict[str, object],
) -> dict[str, Any]:
    return {
        "run_id": run_output.run_id if run_output else None,
        "run_dir": str(run_output.run_dir) if run_output else None,
        "run_manifest": str(run_output.manifest_path) if run_output else None,
        "documents": documents,
        "predictions": predictions,
        "output_dir": str(output_dir),
        "zip": str(zip_path) if zip_path else None,
        "trace_count": traces,
        "strict_validation": strict_validation,
        "relations_enabled": pipeline_options.enable_relations,
        "relation_validation_enabled": pipeline_options.enable_relation_kg_validation,
        "assertion_policy": assertion_policy,
        "candidate_policy": candidate_policy,
        "mode": mode,
        "internal_predictions": (
            str(internal_predictions_path) if internal_predictions_path else None
        ),
        "traces": str(traces_path) if traces_path else None,
        "runtime_metrics": (
            str(runtime_metrics_path) if runtime_metrics_path else None
        ),
        "batch_runtime": batch_runtime,
        "issue_count": len(issues),
        "issues": issues[:20],
    }


def _mapping(value: Any, key: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return {str(item_key): item_value for item_key, item_value in value.items()}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _required_str(value: Any, key: str, parser: argparse.ArgumentParser) -> str:
    text = _optional_str(value)
    if text is None:
        parser.error(f"{key} is required, either in --config or CLI.")
    return text


def _int_setting(cli_value: int | None, config_value: Any, default: int, key: str) -> int:
    value = (
        cli_value
        if cli_value is not None
        else config_value
        if config_value is not None
        else default
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return value


def _export_policy(value: Any, key: str) -> Phase1ExportPolicy:
    policy = str(value or "pipeline").strip().lower()
    if policy not in {"empty", "pipeline", "selective"}:
        raise ValueError(f"{key} must be one of: empty, pipeline, selective.")
    return cast(Phase1ExportPolicy, policy)


def _bool_setting(value: Any, default: bool, key: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _validate_submission_mode(
    mode: str,
    assertion_policy: Phase1ExportPolicy,
    candidate_policy: Phase1ExportPolicy,
    options: PipelineOptions,
    selective_config: Phase1SelectiveExportConfig | None,
) -> None:
    if mode not in {"entity_only", "selective", "full", "custom"}:
        raise ValueError("mode must be one of: entity_only, selective, full, custom.")
    if mode == "custom":
        return
    if mode == "selective":
        if assertion_policy != "selective" or candidate_policy != "selective":
            raise ValueError("selective mode requires selective assertion and candidate policies.")
        if selective_config is None:
            raise ValueError("selective mode requires a selective config block.")
        if not options.enable_context:
            raise ValueError("selective mode requires context classification.")
        if selective_config.candidate_enabled and not options.enable_linking:
            non_pinned_sources = sorted(
                source
                for _, source in selective_config.candidate_source_thresholds
                if not source.startswith("dictionary_")
            )
            if non_pinned_sources:
                raise ValueError(
                    "selective candidate export without linking only supports pinned "
                    f"dictionary sources, got: {non_pinned_sources}"
                )
        return
    if mode == "entity_only":
        if assertion_policy != "empty" or candidate_policy != "empty":
            raise ValueError("entity_only mode requires empty assertion and candidate policies.")
        enabled = {
            "enable_context": options.enable_context,
            "enable_linking": options.enable_linking,
            "enable_candidate_reranking": options.enable_candidate_reranking,
            "enable_entity_kg_validation": options.enable_entity_kg_validation,
            "enable_relations": options.enable_relations,
            "enable_relation_kg_validation": options.enable_relation_kg_validation,
        }
        active = sorted(name for name, value in enabled.items() if value)
        if active:
            raise ValueError(f"entity_only mode requires disabled downstream stages: {active}")
        return
    if assertion_policy != "pipeline" or candidate_policy != "pipeline":
        raise ValueError("full mode requires pipeline assertion and candidate policies.")
    required = {
        "enable_context": options.enable_context,
        "enable_linking": options.enable_linking,
        "enable_candidate_reranking": options.enable_candidate_reranking,
        "enable_entity_kg_validation": options.enable_entity_kg_validation,
    }
    inactive = sorted(name for name, value in required.items() if not value)
    if inactive:
        raise ValueError(f"full mode requires enabled assertion/linking stages: {inactive}")


def _selective_config(
    config: dict[str, Any],
    assertion_policy: Phase1ExportPolicy,
    candidate_policy: Phase1ExportPolicy,
) -> Phase1SelectiveExportConfig | None:
    if "selective" not in {assertion_policy, candidate_policy}:
        return None
    payload = _mapping(config.get("selective"), "selective")
    assertions = _mapping(payload.get("assertions"), "selective.assertions")
    candidates = _mapping(payload.get("candidates"), "selective.candidates")
    evidence_path = _optional_str(assertions.get("calibrated_evidence_map"))
    calibrated_evidence = (
        load_calibrated_assertion_map(evidence_path)
        if evidence_path
        else frozenset()
    )
    reviewed_path = _optional_str(candidates.get("reviewed_map"))
    reviewed = load_reviewed_candidate_map(reviewed_path) if reviewed_path else frozenset()
    return Phase1SelectiveExportConfig.from_mapping(
        payload,
        reviewed_candidates=reviewed,
        calibrated_assertion_evidence=calibrated_evidence,
    )


if __name__ == "__main__":
    main()
