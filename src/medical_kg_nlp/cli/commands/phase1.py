"""Phase 1 benchmark plugin command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from medical_kg_nlp.benchmarks.phase1.model_dataset import (
    Phase1ModelDatasetConfig,
    build_phase1_model_dataset,
)
from medical_kg_nlp.benchmarks.phase1.model_selection import (
    Phase1ModelSelectionConfig,
    calibrate_phase1_model_thresholds,
    compare_phase1_ner_variants,
    load_internal_predictions,
    write_phase1_model_selection_report,
)
from medical_kg_nlp.benchmarks.phase1.round2 import (
    build_phase1_round2_audit,
    write_phase1_round2_audit,
)
from medical_kg_nlp.benchmarks.phase1.runner import (
    BenchmarkExportPolicy,
    Phase1BenchmarkConfig,
    run_phase1_benchmark,
)
from medical_kg_nlp.mining.io import load_documents, write_json
from medical_kg_nlp.pipeline.parallel_batch import ParallelBackend
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.run_output import create_hashed_run_dir, path_in_run

__all__ = [
    "audit_phase1_round2",
    "build_phase1_model_data",
    "calibrate_phase1_model_data",
    "compare_phase1_model_variants",
    "run_phase1_submission",
]


def run_phase1_submission(args: argparse.Namespace) -> int:
    """Build, strict-validate, and archive a Phase 1 artifact."""

    run_output = None
    if args.run_root:
        if Path(args.output_dir).is_absolute() or Path(args.zip).is_absolute():
            raise ValueError("Hashed Phase 1 output and ZIP paths must be relative")
        source_input = args.input_dir or args.documents
        provenance_inputs = [
            source_input,
            args.pipeline_config or "pipeline-config:none",
            args.dictionary,
            args.abbreviations,
            *args.validation_dictionaries,
            *args.provenance_input,
        ]
        run_output = create_hashed_run_dir(
            args.run_root,
            label=args.run_label,
            inputs=provenance_inputs,
            resolved_config={
                "source_archive_sha256": args.source_archive_sha256,
                "assertion_policy": args.assertion_policy,
                "candidate_policy": args.candidate_policy,
                "max_candidates": args.max_candidates,
                "parallel_backend": args.parallel_backend,
                "workers": args.workers,
                "chunksize": args.chunksize,
            },
        )
    output_dir = path_in_run(args.output_dir, run_output)
    zip_path = path_in_run(args.zip, run_output)
    report = run_phase1_benchmark(
        Phase1BenchmarkConfig(
            input_dir=None if args.input_dir is None else Path(args.input_dir),
            output_dir=output_dir,
            zip_path=zip_path,
            dictionary_path=Path(args.dictionary),
            abbreviation_path=Path(args.abbreviations),
            documents_path=None if args.documents is None else Path(args.documents),
            expected_source_archive_sha256=args.source_archive_sha256,
            pipeline_config_path=(
                Path(args.pipeline_config) if args.pipeline_config else None
            ),
            validation_dictionary_paths=tuple(
                Path(path) for path in args.validation_dictionaries
            ),
            assertion_policy=cast(BenchmarkExportPolicy, args.assertion_policy),
            candidate_policy=cast(BenchmarkExportPolicy, args.candidate_policy),
            max_candidates=args.max_candidates,
            backend=cast(ParallelBackend, args.parallel_backend),
            workers=args.workers,
            chunksize=args.chunksize,
        )
    )
    if run_output is not None:
        manifest = json.loads(run_output.manifest_path.read_text(encoding="utf-8"))
        manifest["benchmark"] = report
        manifest["outputs"] = {
            "directory": str(output_dir),
            "zip": str(zip_path),
            "zip_sha256": sha256_file(zip_path),
        }
        write_json(run_output.manifest_path, manifest)
        report["run_id"] = run_output.run_id
        report["run_dir"] = str(run_output.run_dir)
        report["run_manifest"] = str(run_output.manifest_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def audit_phase1_round2(args: argparse.Namespace) -> int:
    """Write audit-only Round 2 distribution and overlap evidence."""

    documents_path = Path(args.documents)
    audit = build_phase1_round2_audit(
        load_documents(documents_path),
        reference_input_dir=Path(args.reference_input_dir),
        reference_gold_dir=Path(args.reference_gold_dir),
        reference_split_manifest=Path(args.reference_split_manifest),
    )
    manifest = write_phase1_round2_audit(
        audit,
        Path(args.output_dir),
        documents_manifest_path=documents_path,
    )
    summary = {
        "document_count": audit["profile"]["documents"]["count"],
        "novelty_document_count": len(audit["novelty_queue"]),
        "runtime_eligible": False,
        "manifest": manifest,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_phase1_model_data(args: argparse.Namespace) -> int:
    """Build the five-type NER view from the frozen manual-gold train split."""

    report = build_phase1_model_dataset(
        Path(args.output_dir),
        config=Phase1ModelDatasetConfig(
            input_dir=Path(args.input_dir),
            gold_dir=Path(args.gold_dir),
            frozen_split_manifest=Path(args.frozen_split_manifest),
            public_spec_input=Path(args.public_spec_input),
            public_spec_expected=Path(args.public_spec_expected),
            development_fraction=args.development_fraction,
            split_salt=args.split_salt,
            max_characters=args.max_characters,
            include_empty_chunks=not args.exclude_empty_chunks,
            empty_chunk_rate=args.empty_chunk_rate,
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def calibrate_phase1_model_data(args: argparse.Namespace) -> int:
    """Calibrate model thresholds without reading frozen holdout labels."""

    config = _model_selection_config(args)
    predictions_path = Path(args.pred)
    report = calibrate_phase1_model_thresholds(
        load_internal_predictions(predictions_path),
        config=config,
        prediction_path=predictions_path,
    )
    write_phase1_model_selection_report(report, Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def compare_phase1_model_variants(args: argparse.Namespace) -> int:
    """Rank exactly three entity compositions and optionally open the holdout gate."""

    variants: dict[str, Path] = {}
    for raw in args.variant:
        name, separator, value = str(raw).partition("=")
        if not separator or not name or not value:
            raise ValueError("--variant must use NAME=DIR_OR_ZIP")
        if name in variants:
            raise ValueError(f"Duplicate --variant name {name!r}")
        variants[name] = Path(value)
    report = compare_phase1_ner_variants(
        variants,
        config=_model_selection_config(args),
        open_frozen_holdout=bool(args.open_frozen_holdout),
    )
    write_phase1_model_selection_report(report, Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _model_selection_config(args: argparse.Namespace) -> Phase1ModelSelectionConfig:
    thresholds = getattr(args, "thresholds", None)
    if thresholds is None:
        return Phase1ModelSelectionConfig(
            input_dir=Path(args.input_dir),
            gold_dir=Path(args.gold_dir),
            model_split_manifest=Path(args.model_split_manifest),
            frozen_split_manifest=Path(args.frozen_split_manifest),
        )
    return Phase1ModelSelectionConfig(
        input_dir=Path(args.input_dir),
        gold_dir=Path(args.gold_dir),
        model_split_manifest=Path(args.model_split_manifest),
        frozen_split_manifest=Path(args.frozen_split_manifest),
        threshold_grid=tuple(
            sorted(set(float(value) for value in thresholds))
        )
    )
