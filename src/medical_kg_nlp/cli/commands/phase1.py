"""Phase 1 benchmark plugin command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from medical_kg_nlp.benchmarks.phase1.runner import (
    BenchmarkExportPolicy,
    Phase1BenchmarkConfig,
    run_phase1_benchmark,
)
from medical_kg_nlp.pipeline.parallel_batch import ParallelBackend

__all__ = ["run_phase1"]


def run_phase1(args: argparse.Namespace) -> int:
    """Build, strict-validate, and archive a Phase 1 artifact."""

    report = run_phase1_benchmark(
        Phase1BenchmarkConfig(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            zip_path=Path(args.zip),
            dictionary_path=Path(args.dictionary),
            abbreviation_path=Path(args.abbreviations),
            pipeline_config_path=(
                Path(args.pipeline_config) if args.pipeline_config else None
            ),
            assertion_policy=cast(BenchmarkExportPolicy, args.assertion_policy),
            candidate_policy=cast(BenchmarkExportPolicy, args.candidate_policy),
            max_candidates=args.max_candidates,
            backend=cast(ParallelBackend, args.parallel_backend),
            workers=args.workers,
            chunksize=args.chunksize,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
