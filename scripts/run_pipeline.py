#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline.parallel_batch import ParallelBatchOptions, run_batch_with_trace_parallel
from medical_kg_nlp.utils.io import write_jsonl
from medical_kg_nlp.utils.run_output import create_hashed_run_dir, path_in_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run medical KG NLP pipeline.")
    parser.add_argument("--input", required=True, help="Input JSONL with document_id and text.")
    parser.add_argument("--output", required=True, help="Output predictions JSONL.")
    parser.add_argument(
        "--run-root",
        help="Optional root for hashed run directories. Relative output paths are written under it.",
    )
    parser.add_argument("--run-label", default="pipeline", help="Label embedded in the hashed run directory.")
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
    run_output = (
        create_hashed_run_dir(
            args.run_root,
            label=args.run_label,
            inputs=[args.input, args.dictionary, args.abbreviations],
        )
        if args.run_root
        else None
    )
    output_path = path_in_run(args.output, run_output)
    write_jsonl(output_path, predictions)
    if run_output:
        print(
            json.dumps(
                {
                    "run_id": run_output.run_id,
                    "run_dir": str(run_output.run_dir),
                    "run_manifest": str(run_output.manifest_path),
                    "output": str(output_path),
                    "prediction_count": len(predictions),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
