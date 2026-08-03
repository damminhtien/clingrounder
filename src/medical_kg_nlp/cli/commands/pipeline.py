"""Pipeline run command orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.pipeline.config_loader import ResolvedPipelineConfig
from medical_kg_nlp.pipeline.factory import PipelineFactoryConfig
from medical_kg_nlp.pipeline.parallel_batch import (
    ParallelBatchOptions,
    run_batch_with_trace_parallel,
)
from medical_kg_nlp.utils.io import write_jsonl
from medical_kg_nlp.utils.run_output import (
    collect_git_metadata,
    create_hashed_run_dir,
    path_in_run,
)

__all__ = ["inspect_pipeline_config", "run_pipeline"]


def inspect_pipeline_config(args: argparse.Namespace) -> int:
    """Print one resolved profile without constructing heavyweight components."""

    report = ResolvedPipelineConfig.load(
        args.config,
        require_profile=True,
    ).inspection_report()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.check_resources and any(
        not resource["exists"] for resource in report["resources"]
    ):
        return 1
    return 0


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
    manifest_path = (
        path_in_run(args.manifest, run_output)
        if args.manifest
        else output_path.with_suffix(output_path.suffix + ".manifest.json")
    )
    manifest = _run_manifest(
        args=args,
        resolved=ResolvedPipelineConfig.load(args.config, require_profile=True),
        input_path=Path(args.input),
        output_path=output_path,
        prediction_count=len(predictions),
        hashed_run_manifest=(
            None if run_output is None else str(run_output.manifest_path)
        ),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "manifest": str(manifest_path),
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
    config = ResolvedPipelineConfig.load(
        args.config,
        require_profile=True,
    ).factory_config
    return replace(
        config,
        recognition_dictionary_path=args.dictionary
        or config.recognition_dictionary_path,
        abbreviation_path=args.abbreviations or config.abbreviation_path,
    )


def _run_manifest(
    *,
    args: argparse.Namespace,
    resolved: ResolvedPipelineConfig,
    input_path: Path,
    output_path: Path,
    prediction_count: int,
    hashed_run_manifest: str | None,
) -> dict[str, object]:
    """Capture enough identity to replay a public toolkit run on another machine."""

    return {
        "schema_version": "medical-kg.pipeline-run.v1",
        "profile": resolved.inspection_report(),
        "input": _file_artifact(input_path),
        "output": _file_artifact(output_path),
        "prediction_count": prediction_count,
        "execution": {
            "parallel_backend": args.parallel_backend,
            "workers": args.workers,
            "chunksize": args.chunksize,
            "fail_fast": not args.no_fail_fast,
        },
        "source_control": collect_git_metadata(),
        "hashed_run_manifest": hashed_run_manifest,
    }


def _file_artifact(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
