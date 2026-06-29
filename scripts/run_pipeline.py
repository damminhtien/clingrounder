#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline.runner import PipelineRunner
from medical_kg_nlp.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run medical KG NLP pipeline.")
    parser.add_argument("--input", required=True, help="Input JSONL with document_id and text.")
    parser.add_argument("--output", required=True, help="Output predictions JSONL.")
    parser.add_argument("--dictionary", default="data/dictionaries/seed_concepts.jsonl")
    parser.add_argument("--abbreviations", default="data/dictionaries/abbreviations.jsonl")
    args = parser.parse_args()

    adapter = SyntheticDatasetAdapter()
    documents = adapter.load_documents(args.input)
    runner = PipelineRunner(dictionary_path=args.dictionary, abbreviation_path=args.abbreviations)
    predictions = [runner.process_document(document).to_json() for document in documents]
    write_jsonl(args.output, predictions)


if __name__ == "__main__":
    main()
