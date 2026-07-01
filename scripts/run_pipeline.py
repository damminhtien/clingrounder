#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline.parallel_batch import ParallelBatchOptions, run_batch_with_trace_parallel
from medical_kg_nlp.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run medical KG NLP pipeline.")
    parser.add_argument("--input", required=True, help="Input JSONL with document_id and text.")
    parser.add_argument("--output", required=True, help="Output predictions JSONL.")
    parser.add_argument("--dictionary", default="data/dictionaries/seed_concepts.jsonl")
    parser.add_argument("--abbreviations", default="data/dictionaries/abbreviations.jsonl")
    parser.add_argument(
        "--parallel-backend",
        choices=("serial", "thread", "process"),
        default="process",
        help="Batch execution backend. With --workers 1 this still runs serially.",
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
    documents = adapter.load_documents(args.input)
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
    predictions = [result.prediction.to_json() for result in run_results]
    write_jsonl(args.output, predictions)


if __name__ == "__main__":
    main()
