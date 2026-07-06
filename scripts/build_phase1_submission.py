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
    load_phase1_text_documents,
    validate_phase1_submission_dir,
    validate_phase1_submission_zip,
    write_phase1_output_dir,
    zip_phase1_output_dir,
)
from medical_kg_nlp.pipeline.parallel_batch import ParallelBackend, ParallelBatchOptions, run_batch_with_trace_parallel
from medical_kg_nlp.pipeline.options import PipelineOptions
from medical_kg_nlp.utils.io import read_yaml
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
    parser.add_argument("--pred", help="Optional internal prediction JSONL to export instead of running.")
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
    output_dir_arg = _required_str(args.output_dir or config.get("output_dir"), "output_dir", parser)
    zip_arg = _optional_str(args.zip_path if args.zip_path is not None else config.get("zip"))
    run_root = _optional_str(args.run_root if args.run_root is not None else config.get("run_root"))
    run_label = _required_str(args.run_label or config.get("run_label") or "phase1", "run_label", parser)
    pred_path = _optional_str(args.pred if args.pred is not None else config.get("pred"))
    dictionary_path = _required_str(
        args.dictionary or config.get("dictionary") or "data/dictionaries/seed_concepts.jsonl",
        "dictionary",
        parser,
    )
    abbreviation_path = _required_str(
        args.abbreviations or config.get("abbreviations") or "data/dictionaries/abbreviations.jsonl",
        "abbreviations",
        parser,
    )
    max_candidates = _int_setting(args.max_candidates, config.get("max_candidates"), 1, "max_candidates")
    expected_count = _int_setting(args.expected_count, config.get("expected_count"), 100, "expected_count")
    parallel_backend = str(args.parallel_backend or parallel_config.get("backend") or "process")
    if parallel_backend not in {"serial", "thread", "process"}:
        raise ValueError("parallel.backend must be one of: serial, thread, process.")
    workers = _int_setting(args.workers, parallel_config.get("workers"), 1, "parallel.workers")
    chunksize = _int_setting(args.chunksize, parallel_config.get("chunksize"), 4, "parallel.chunksize")
    fail_fast = False if args.no_fail_fast else _bool_setting(parallel_config.get("fail_fast"), True, "parallel.fail_fast")
    strict_validation = (
        False
        if args.no_strict_validation
        else _bool_setting(config.get("strict_validation"), True, "strict_validation")
    )
    pipeline_options = PipelineOptions.from_mapping(pipeline_config)

    run_output = (
        create_hashed_run_dir(
            run_root,
            label=run_label,
            inputs=[input_dir, dictionary_path, pred_path or "pipeline"],
        )
        if run_root
        else None
    )
    documents = load_phase1_text_documents(input_dir)
    if pred_path:
        predictions = SyntheticDatasetAdapter().load_gold(pred_path)
        traces = []
    else:
        run_results = run_batch_with_trace_parallel(
            documents,
            dictionary_path=dictionary_path,
            abbreviation_path=abbreviation_path,
            pipeline_options=pipeline_options,
            parallel_options=ParallelBatchOptions(
                backend=cast(ParallelBackend, parallel_backend),
                max_workers=workers,
                chunksize=chunksize,
                fail_fast=fail_fast,
            ),
        )
        predictions = [result.prediction for result in run_results]
        traces = [result.trace.to_json() for result in run_results]

    output_dir = path_in_run(output_dir_arg, run_output)
    source_text_by_document = {document.document_id: document.text for document in documents}
    write_phase1_output_dir(
        predictions,
        output_dir,
        max_candidates=max_candidates,
        source_text_by_document=source_text_by_document,
    )
    dictionary = DictionaryStore.from_jsonl(dictionary_path)
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


def _required_str(value: Any, key: str, parser: argparse.ArgumentParser) -> str:
    text = _optional_str(value)
    if text is None:
        parser.error(f"{key} is required, either in --config or CLI.")
    return text


def _int_setting(cli_value: int | None, config_value: Any, default: int, key: str) -> int:
    value = cli_value if cli_value is not None else config_value if config_value is not None else default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return value


def _bool_setting(value: Any, default: bool, key: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


if __name__ == "__main__":
    main()
