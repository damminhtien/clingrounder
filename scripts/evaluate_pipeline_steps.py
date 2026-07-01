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
from medical_kg_nlp.pipeline.runner import PipelineRunner
from medical_kg_nlp.utils.io import write_jsonl


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
    parser.add_argument("--pred", help="Optional existing prediction JSONL to evaluate.")
    parser.add_argument(
        "--reference-gold",
        help="Optional reference/train gold JSONL for unseen-code overlap profiling.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Maximum profile top-list rows.")
    args = parser.parse_args()

    adapter = SyntheticDatasetAdapter()
    documents = adapter.load_documents(args.documents)
    gold = adapter.load_gold(args.gold)
    reference_gold = adapter.load_gold(args.reference_gold) if args.reference_gold else None
    dictionary = DictionaryStore.from_jsonl(args.dictionary)
    output_dir = Path(args.output_dir)

    if args.pred:
        predictions = adapter.load_gold(args.pred)
        traces = []
    else:
        runner = PipelineRunner(
            dictionary_path=args.dictionary,
            abbreviation_path=args.abbreviations,
        )
        run_results = [runner.process_document_with_trace(document) for document in documents]
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
    write_pipeline_report(report, output_dir)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
