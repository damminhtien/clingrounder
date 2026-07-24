"""Pipeline run command orchestration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline.config_loader import ResolvedPipelineConfig
from medical_kg_nlp.pipeline.factory import PipelineFactoryConfig
from medical_kg_nlp.pipeline.parallel_batch import (
    ParallelBatchOptions,
    run_batch_with_trace_parallel,
)
from medical_kg_nlp.utils.io import write_jsonl
from medical_kg_nlp.utils.run_output import create_hashed_run_dir, path_in_run

__all__ = ["run_pipeline"]


def run_pipeline(args: argparse.Namespace) -> int:
    """Load documents, run configured components, and write prediction JSONL."""

    adapter = SyntheticDatasetAdapter()
    documents = adapter.load_documents(args.input)
    factory_config = _factory_config(args)
    results = run_batch_with_trace_parallel(
        documents,
        factory_config=factory_config,
        parallel_options=ParallelBatchOptions(
            backend=args.parallel_backend,
            max_workers=args.workers,
            chunksize=args.chunksize,
            fail_fast=not args.no_fail_fast,
        ),
    )
    predictions = [result.prediction.to_json() for result in results]
    run_output = (
        create_hashed_run_dir(
            args.run_root,
            label=args.run_label,
            inputs=[args.input, args.config or "", args.dictionary or ""],
            resolved_config=vars(args),
        )
        if args.run_root
        else None
    )
    output_path = path_in_run(args.output, run_output)
    write_jsonl(output_path, predictions)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "prediction_count": len(predictions),
                "run_id": run_output.run_id if run_output else None,
                "run_dir": str(run_output.run_dir) if run_output else None,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _factory_config(args: argparse.Namespace) -> PipelineFactoryConfig:
    if args.config:
        config = ResolvedPipelineConfig.load(args.config).factory_config
    else:
        config = PipelineFactoryConfig()
    return replace(
        config,
        recognition_dictionary_path=args.dictionary
        or config.recognition_dictionary_path,
        abbreviation_path=args.abbreviations or config.abbreviation_path,
    )
