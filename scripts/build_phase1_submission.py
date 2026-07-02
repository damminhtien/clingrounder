#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from medical_kg_nlp.pipeline.parallel_batch import ParallelBatchOptions, run_batch_with_trace_parallel
from medical_kg_nlp.utils.run_output import create_hashed_run_dir, path_in_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase 1 flat JSON files and optional output.zip from input TXT files.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing 1.txt..100.txt.")
    parser.add_argument("--output-dir", required=True, help="Directory that will contain 1.json..100.json.")
    parser.add_argument("--zip", dest="zip_path", help="Optional submission zip path.")
    parser.add_argument(
        "--run-root",
        help="Optional root for hashed run directories. Relative output paths are written under it.",
    )
    parser.add_argument("--run-label", default="phase1", help="Label embedded in the hashed run directory.")
    parser.add_argument("--pred", help="Optional internal prediction JSONL to export instead of running.")
    parser.add_argument(
        "--dictionary",
        default="data/dictionaries/seed_concepts.jsonl",
        help="Dictionary JSONL used for pipeline linking and candidate validation.",
    )
    parser.add_argument(
        "--abbreviations",
        default="data/dictionaries/abbreviations.jsonl",
        help="Abbreviation JSONL used when running the pipeline.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=1,
        help="Max candidates exported per codable Phase 1 entity. Default keeps candidate sets precise.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=100,
        help="Expected JSON files in the official ZIP structure.",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=("serial", "thread", "process"),
        default="process",
        help="Pipeline execution backend when --pred is omitted.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of document workers.")
    parser.add_argument("--chunksize", type=int, default=4, help="Document chunksize for process workers.")
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Collect document worker errors before raising a batch error.",
    )
    args = parser.parse_args()

    run_output = (
        create_hashed_run_dir(
            args.run_root,
            label=args.run_label,
            inputs=[args.input_dir, args.dictionary, args.pred or "pipeline"],
        )
        if args.run_root
        else None
    )
    documents = load_phase1_text_documents(args.input_dir)
    if args.pred:
        predictions = SyntheticDatasetAdapter().load_gold(args.pred)
        traces = []
    else:
        run_results = run_batch_with_trace_parallel(
            documents,
            dictionary_path=args.dictionary,
            abbreviation_path=args.abbreviations,
            parallel_options=ParallelBatchOptions(
                backend=args.parallel_backend,
                max_workers=args.workers,
                chunksize=args.chunksize,
                fail_fast=not args.no_fail_fast,
            ),
        )
        predictions = [result.prediction for result in run_results]
        traces = [result.trace.to_json() for result in run_results]

    output_dir = path_in_run(args.output_dir, run_output)
    write_phase1_output_dir(predictions, output_dir, max_candidates=args.max_candidates)
    zip_path = path_in_run(args.zip_path, run_output) if args.zip_path else None
    if args.zip_path:
        zip_phase1_output_dir(output_dir, zip_path or args.zip_path)

    dictionary = DictionaryStore.from_jsonl(args.dictionary)
    issues = [
        issue.to_json()
        for issue in validate_phase1_submission_dir(
            args.input_dir,
            output_dir,
            dictionary=dictionary,
        )
    ]
    if args.zip_path:
        issues.extend(
            issue.to_json()
            for issue in validate_phase1_submission_zip(
                zip_path or args.zip_path,
                expected_count=args.expected_count,
            )
        )

    summary = {
        "run_id": run_output.run_id if run_output else None,
        "run_dir": str(run_output.run_dir) if run_output else None,
        "run_manifest": str(run_output.manifest_path) if run_output else None,
        "documents": len(documents),
        "predictions": len(predictions),
        "output_dir": str(output_dir),
        "zip": str(zip_path) if zip_path else args.zip_path,
        "trace_count": len(traces),
        "issue_count": len(issues),
        "issues": issues[:20],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
