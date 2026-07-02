#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.evaluation.pipeline_report import build_pipeline_report, write_pipeline_report
from medical_kg_nlp.pipeline.parallel_batch import ParallelBatchOptions, run_batch_with_trace_parallel
from medical_kg_nlp.utils.io import write_jsonl
from medical_kg_nlp.utils.run_output import create_hashed_run_dir, path_in_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stage-wise metrics, validation, trace, and error-analysis reports.",
    )
    parser.add_argument("--documents", required=True, help="Source documents JSONL.")
    parser.add_argument("--gold", required=True, help="Gold annotations in internal prediction JSONL.")
    parser.add_argument(
        "--dictionary",
        default="data/dictionaries/seed_concepts.jsonl",
        help="Dictionary JSONL for linking and validation.",
    )
    parser.add_argument(
        "--abbreviations",
        default="data/dictionaries/abbreviations.jsonl",
        help="Abbreviation JSONL used when this command runs the pipeline.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for report artifacts.")
    parser.add_argument(
        "--run-root",
        help="Optional root for hashed run directories. Relative output paths are written under it.",
    )
    parser.add_argument("--run-label", default="pipeline-report", help="Label embedded in the hashed run directory.")
    parser.add_argument("--pred", help="Optional existing prediction JSONL to evaluate.")
    parser.add_argument(
        "--reference-gold",
        help="Optional reference/train gold JSONL for unseen-code overlap profiling.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Maximum profile top-list rows.")
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

    adapter = SyntheticDatasetAdapter()
    documents = adapter.load_documents(args.documents)
    gold = adapter.load_gold(args.gold)
    reference_gold = adapter.load_gold(args.reference_gold) if args.reference_gold else None
    dictionary = DictionaryStore.from_jsonl(args.dictionary)
    run_output = (
        create_hashed_run_dir(
            args.run_root,
            label=args.run_label,
            inputs=[args.documents, args.gold, args.pred or "pipeline", args.dictionary],
        )
        if args.run_root
        else None
    )
    output_dir = path_in_run(args.output_dir, run_output)

    if args.pred:
        predictions = adapter.load_gold(args.pred)
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
        traces = [result.trace for result in run_results]
        write_jsonl(output_dir / "predictions.jsonl", [prediction.to_json() for prediction in predictions])

    report = build_pipeline_report(
        documents=documents,
        gold=gold,
        predictions=predictions,
        traces=traces,
        dictionary=dictionary,
        reference_gold=reference_gold,
        top_k=args.top_k,
    )
    if run_output:
        report["summary"]["run_id"] = run_output.run_id
        report["summary"]["run_dir"] = str(run_output.run_dir)
        report["summary"]["run_manifest"] = str(run_output.manifest_path)
    write_pipeline_report(report, output_dir)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
