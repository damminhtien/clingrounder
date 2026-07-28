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
from medical_kg_nlp.benchmarks.phase1.model_runtime import (
    run_phase1_model_calibration,
)
from medical_kg_nlp.benchmarks.phase1.model_region_augmentation import (
    Phase1RegionAugmentationConfig,
    build_phase1_region_augmented_dataset,
)
from medical_kg_nlp.benchmarks.phase1.model_selection import (
    Phase1ModelSelectionConfig,
    compare_phase1_ner_variants,
    write_phase1_model_selection_report,
)
from medical_kg_nlp.benchmarks.phase1.round2 import (
    build_phase1_round2_audit,
    load_phase1_round2_documents,
    write_phase1_round2_audit,
)
from medical_kg_nlp.benchmarks.phase1.round2_golden import (
    build_phase1_round2_golden,
    write_phase1_round2_golden,
)
from medical_kg_nlp.benchmarks.phase1.round2_probes import (
    CandidateProbePolicy,
    Phase1Round2ProbeConfig,
    run_phase1_round2_probes,
)
from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import (
    load_phase1_output_source,
)
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.benchmarks.phase1.synthetic_training import (
    Phase1SyntheticTrainingConfig,
    build_phase1_synthetic_training_dataset,
)
from medical_kg_nlp.benchmarks.phase1.qwen_dataset import (
    Phase1QwenDatasetConfig,
    build_phase1_qwen_instruction_dataset,
)
from medical_kg_nlp.benchmarks.phase1.qwen_run_spec import (
    load_phase1_qwen_run_spec,
)
from medical_kg_nlp.benchmarks.phase1.qwen_runner import (
    Phase1QwenProposalRunConfig,
    run_phase1_qwen_proposals,
)
from medical_kg_nlp.benchmarks.phase1.vietnamese_support import (
    build_phase1_vietnamese_model_support,
    load_phase1_vietnamese_support_spec,
)
from medical_kg_nlp.benchmarks.phase1.runner import (
    BenchmarkExportPolicy,
    Phase1BenchmarkConfig,
    run_phase1_benchmark,
)
from medical_kg_nlp.mining.io import load_documents, write_json
from medical_kg_nlp.pipeline.parallel_batch import ParallelBackend
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.run_output import create_hashed_run_dir, path_in_run

__all__ = [
    "audit_phase1_round2",
    "augment_phase1_model_regions",
    "augment_phase1_model_user_synthetic",
    "build_phase1_model_data",
    "build_phase1_qwen_data",
    "build_phase1_round2_golden_command",
    "calibrate_phase1_model_data",
    "compare_phase1_model_variants",
    "inspect_phase1_qwen_run",
    "propose_phase1_qwen_entities",
    "propose_phase1_vietnamese_support",
    "run_phase1_round2_probe_suite",
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


def run_phase1_round2_probe_suite(args: argparse.Namespace) -> int:
    """Build strict probe variants around one frozen Round 2 artifact."""

    report = run_phase1_round2_probes(
        Phase1Round2ProbeConfig(
            documents_path=Path(args.documents),
            expected_source_archive_sha256=args.source_archive_sha256,
            base=Path(args.base),
            expected_base_sha256=args.expected_base_sha256,
            dictionary_paths=(
                Path(args.dictionary),
                *(Path(path) for path in args.validation_dictionaries),
            ),
            proposal_sources=tuple(_named_paths(args.source)),
            output_root=Path(args.output_root),
            run_label=args.run_label,
            expected_count=args.expected_count,
            minimum_agreement_sources=args.minimum_agreement_sources,
            expand_repeated_mentions=not args.no_expand_repeated_mentions,
            full_source_names=tuple(args.build_full_source),
            consensus_source_names=tuple(args.build_consensus_source),
            candidate_probe_policies=tuple(
                cast(CandidateProbePolicy, policy) for policy in args.candidate_probe
            ),
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_phase1_round2_golden_command(args: argparse.Namespace) -> int:
    """Build non-official strict and review labels from independent Round 2 sources."""

    named_sources = _named_paths(args.source)
    source_names = [name for name, _ in named_sources]
    if len(named_sources) < 2 or len(source_names) != len(set(source_names)):
        raise ValueError("--source requires at least two uniquely named proposal artifacts")
    dictionary_paths = (
        Path(args.dictionary),
        *(Path(path) for path in args.validation_dictionaries),
    )
    dictionary_entries = [
        entry
        for path in dictionary_paths
        for entry in DictionaryStore.load_entries_jsonl(path)
    ]
    dictionary = DictionaryStore(dictionary_entries)
    mined_documents = load_phase1_round2_documents(
        load_documents(args.documents),
        expected_archive_sha256=args.source_archive_sha256,
        expected_count=args.expected_count,
    )
    documents = [
        ClinicalDocument(document_id=document.document_id, text=document.text)
        for document in mined_documents
    ]
    source_text_by_doc = {document.document_id: document.text for document in documents}
    proposal_sources = {
        name: load_phase1_output_source(path) for name, path in named_sources
    }
    btc_example_input = Path(args.btc_example_input)
    btc_example_output = Path(args.btc_example_output)
    run_output = create_hashed_run_dir(
        args.output_root,
        label=args.run_label,
        inputs=(
            args.documents,
            *(path for _, path in named_sources),
            *dictionary_paths,
            btc_example_input,
            btc_example_output,
        ),
        resolved_config={
            "source_archive_sha256": args.source_archive_sha256,
            "expected_count": args.expected_count,
            "minimum_sources": args.minimum_sources,
            "source_names": source_names,
            "official_gold": False,
        },
    )
    report = build_phase1_round2_golden(
        source_text_by_doc,
        proposal_sources,
        dictionary,
        minimum_sources=args.minimum_sources,
    )
    run_manifest = json.loads(run_output.manifest_path.read_text(encoding="utf-8"))
    manifest = write_phase1_round2_golden(
        report,
        run_output.run_dir,
        documents=documents,
        dictionary=dictionary,
        provenance={
            "source_archive_sha256": args.source_archive_sha256,
            "proposal_sources": {
                name: str(path) for name, path in named_sources
            },
            "input_artifacts": run_manifest["input_artifacts"],
            "btc_example": {
                "input": str(btc_example_input),
                "input_sha256": sha256_file(btc_example_input),
                "output": str(btc_example_output),
                "output_sha256": sha256_file(btc_example_output),
            },
        },
    )
    run_manifest["inferred_gold"] = manifest
    write_json(run_output.manifest_path, run_manifest)
    result = {
        "run_id": run_output.run_id,
        "run_dir": str(run_output.run_dir),
        "manifest": str(run_output.run_dir / "manifest.json"),
        "official_gold": False,
        "summary": report["summary"],
        "artifacts": manifest["artifacts"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
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


def augment_phase1_model_regions(args: argparse.Namespace) -> int:
    """Build train-only Q&A/educational views from reviewed span records."""

    report = build_phase1_region_augmented_dataset(
        Path(args.output_dir),
        config=Phase1RegionAugmentationConfig(
            source_dataset_path=Path(args.source_dataset),
            source_manifest_path=Path(args.source_manifest),
            source_build_manifest_path=Path(args.source_build_manifest),
            max_synthetic_train_fraction=args.max_synthetic_fraction,
            seed=args.seed,
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def augment_phase1_model_user_synthetic(args: argparse.Namespace) -> int:
    """Build a human-development, bounded-synthetic-train span dataset."""

    report = build_phase1_synthetic_training_dataset(
        Phase1SyntheticTrainingConfig(
            archive_path=Path(args.archive),
            expected_archive_sha256=args.archive_sha256,
            human_spans_path=Path(args.source_dataset),
            human_manifest_path=Path(args.source_manifest),
            output_dir=Path(args.output_dir),
            max_synthetic_train_fraction=args.max_synthetic_fraction,
            selection_seed=args.seed,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_phase1_qwen_data(args: argparse.Namespace) -> int:
    """Build extraction, missing-review, and optional hard-negative records."""

    report = build_phase1_qwen_instruction_dataset(
        Phase1QwenDatasetConfig(
            spans_path=Path(args.source_dataset),
            spans_manifest_path=Path(args.source_manifest),
            output_dir=Path(args.output_dir),
            hard_negative_predictions_path=(
                None
                if args.hard_negative_predictions is None
                else Path(args.hard_negative_predictions)
            ),
            include_development=not args.exclude_development,
            review_masks_per_train_record=args.review_masks_per_train_record,
            review_keep_fraction=args.review_keep_fraction,
            review_seed=args.review_seed,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def inspect_phase1_qwen_run(args: argparse.Namespace) -> int:
    """Validate a Qwen run without importing Torch or loading the checkpoint."""

    spec = load_phase1_qwen_run_spec(args.config)
    payload = spec.to_dict()
    payload["config"] = {
        "path": spec.relative_path(spec.config_path),
        "sha256": sha256_file(spec.config_path),
    }
    payload["dataset"]["present"] = (
        spec.dataset_path.is_file() and spec.dataset_manifest_path.is_file()
    )
    payload["commands"] = {
        "prefetch": list(spec.prefetch_command),
        "build_dataset": [
            "medical-kg",
            "benchmark",
            "phase1",
            "model-data",
            "build-qwen",
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def propose_phase1_qwen_entities(args: argparse.Namespace) -> int:
    """Run a pinned Qwen checkpoint over an authorized private document manifest."""

    report = run_phase1_qwen_proposals(
        load_phase1_qwen_run_spec(args.config),
        Phase1QwenProposalRunConfig(
            documents_path=Path(args.documents),
            expected_source_archive_sha256=args.source_archive_sha256,
            output_dir=Path(args.output_dir),
            support_sources=tuple(_named_paths(args.support_source)),
            review_source=(
                None
                if args.review_source is None
                else _named_paths([args.review_source])[0]
            ),
            review_max_rounds=args.review_max_rounds,
            review_only=args.review_only,
            expected_document_count=args.expected_count,
            run_adjudication=not args.no_adjudication,
            resume=args.resume,
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def propose_phase1_vietnamese_support(args: argparse.Namespace) -> int:
    """Run a pinned Vietnamese source-task model as Qwen-only support evidence."""

    report = build_phase1_vietnamese_model_support(
        load_phase1_vietnamese_support_spec(args.config),
        documents_path=args.documents,
        expected_source_archive_sha256=args.source_archive_sha256,
        output_dir=args.output_dir,
        expected_document_count=args.expected_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def calibrate_phase1_model_data(args: argparse.Namespace) -> int:
    """Verify a full checkpoint and calibrate it without opening holdout labels."""

    report = run_phase1_model_calibration(
        args.pipeline_config,
        args.output_dir,
    )
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


def _named_paths(values: list[str]) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for raw in values:
        name, separator, value = str(raw).partition("=")
        if not separator or not name or not value:
            raise ValueError("--source must use NAME=DIR_OR_ZIP")
        paths.append((name, Path(value)))
    return paths
