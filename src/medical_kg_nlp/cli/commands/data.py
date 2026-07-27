"""Thin CLI orchestration for task-neutral data mining operations."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import yaml

from medical_kg_nlp.mining.annotation_manifest import (
    materialize_annotation_manifest,
)
from medical_kg_nlp.mining.abbreviations import (
    benchmark_abbreviation_knowledge,
    build_runtime_abbreviation_table,
    load_abbreviation_mining_policy,
    load_snapshot_splits,
    mine_abbreviations,
)
from medical_kg_nlp.mining.coverage import CoverageCubePlanner, CoverageTarget
from medical_kg_nlp.mining.cooccurrence import (
    load_cooccurrence_policy,
    mine_cooccurrence_relations,
)
from medical_kg_nlp.mining.crosswalk import crosswalk_mentions, load_crosswalk_policies
from medical_kg_nlp.mining.crosswalk_links import (
    load_crosswalk_link_policy,
    load_crosswalk_rows,
    materialize_exact_crosswalk_links,
)
from medical_kg_nlp.mining.dedup import StableTextDeduplicator
from medical_kg_nlp.mining.exact_quote_curriculum import (
    ExactQuoteCurriculumConfig,
    build_exact_quote_curriculum,
)
from medical_kg_nlp.mining.dailymed_product_aliases import (
    build_dailymed_product_alias_proposals,
)
from medical_kg_nlp.mining.curation import (
    curate_annotations,
    load_annotation_curation_policy,
)
from medical_kg_nlp.mining.graph_knowledge import (
    GraphCompilationConfig,
    compile_knowledge_graph,
)
from medical_kg_nlp.mining.fusion import run_corpus_fusion_plan
from medical_kg_nlp.mining.harmonization import (
    harmonize_annotations,
    load_annotation_harmonization_policy,
)
from medical_kg_nlp.mining.io import (
    iter_documents,
    load_annotations,
    load_documents,
    load_relations,
    load_source_artifacts,
    write_json,
    write_jsonl,
    write_text,
)
from medical_kg_nlp.mining.labeling import PolicyAwareProposalLabelerAdapter
from medical_kg_nlp.mining.knowledge import (
    compile_mined_aliases,
    load_alias_promotion_policy,
)
from medical_kg_nlp.mining.lexicon import build_mention_inventory, load_mention_inventory
from medical_kg_nlp.mining.linked_aliases import (
    build_linked_alias_proposals,
    load_linked_alias_policy,
)
from medical_kg_nlp.mining.mappings.dailymed_rxnorm import (
    audit_dailymed_rxnorm_mapping,
    compile_dailymed_rxnorm_mapping,
)
from medical_kg_nlp.mining.mappings.dailymed_product_rxnorm import (
    link_dailymed_products_to_rxnorm,
)
from medical_kg_nlp.mining.mappings.rxnorm_ndc import compile_rxnorm_ndc_index
from medical_kg_nlp.mining.model_dataset import (
    SpanDatasetConfig,
    export_span_dataset,
    load_dataset_splits,
)
from medical_kg_nlp.mining.ontologies import (
    OBOGraphCompilationConfig,
    compile_hpo_associations,
    compile_obo_graph_release,
)
from medical_kg_nlp.mining.policy import SourcePolicyGate
from medical_kg_nlp.mining.ports import ProposalLabelerPort, RelationLabelerPort
from medical_kg_nlp.mining.profile import (
    build_dataset_profile,
    profile_blocking_issue_count,
)
from medical_kg_nlp.mining.recognition_benchmark import (
    benchmark_recognition_dictionary,
)
from medical_kg_nlp.mining.recognition_knowledge import (
    compile_recognition_knowledge,
    load_recognition_knowledge_policy,
)
from medical_kg_nlp.mining.reconciliation import reconcile_exact_duplicates
from medical_kg_nlp.mining.release import (
    build_mining_release_lock,
    verify_mining_release_lock,
)
from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    ReviewStatus,
    SourceRequest,
)
from medical_kg_nlp.mining.quality import GoldAgreementGate, ReviewAgreementEvaluator
from medical_kg_nlp.mining.registry import load_source_registry
from medical_kg_nlp.mining.source_status import (
    load_source_processing_index,
    validate_source_processing_paths,
)
from medical_kg_nlp.mining.review import JsonlReviewBackend
from medical_kg_nlp.mining.runner import (
    artifact_store_from_uri,
    materialize_documents,
    run_mining_plan,
    sync_source,
)
from medical_kg_nlp.mining.section_evidence import (
    attach_block_evidence,
    load_block_evidence_policy,
)
from medical_kg_nlp.mining.snapshot import SnapshotBuilder, SnapshotSplitConfig
from medical_kg_nlp.mining.splits import (
    load_split_document_ids,
    select_mined_records,
    select_mined_records_with_metadata,
)
from medical_kg_nlp.terminology import SQLiteTerminologyRepository
from medical_kg_nlp.mining.storage import materialize_stored_object
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.hashing import sha256_file

__all__ = [
    "build_dataset",
    "build_exact_quote_curriculum_dataset",
    "attach_dataset_block_evidence",
    "build_lexicon",
    "benchmark_recognition_knowledge",
    "crosswalk_lexicon",
    "audit_dailymed_rxnorm",
    "attach_exact_crosswalk_links",
    "compile_dailymed_rxnorm",
    "compile_rxnorm_ndc",
    "compile_graph_knowledge",
    "compile_hpo_association_knowledge",
    "compile_obo_ontology",
    "compile_alias_knowledge",
    "compile_recognition_knowledge_artifact",
    "curate_annotation_dataset",
    "export_review",
    "export_span_training_dataset",
    "freeze_snapshot",
    "fuse_datasets",
    "harmonize_dataset",
    "import_review",
    "inspect_dataset",
    "link_dailymed_products",
    "lock_mining_release",
    "materialize_artifact",
    "mine_abbreviation_knowledge",
    "mine_cooccurrence",
    "propose_labels",
    "propose_dailymed_product_aliases",
    "propose_linked_aliases",
    "propose_relations",
    "reconcile_duplicates",
    "report_coverage",
    "review_quality",
    "run_plan",
    "sync_registered_source",
    "validate_registry",
    "verify_mining_release",
]


def validate_registry(args: argparse.Namespace) -> int:
    registry = load_source_registry(args.registry)
    processing = None
    if args.processing_index:
        processing = load_source_processing_index(args.processing_index)
        path_errors = validate_source_processing_paths(
            processing,
            registry,
            repository_root=args.repository_root,
        )
        if path_errors:
            raise ValueError(
                "Source processing index is not discoverable: " + ", ".join(path_errors)
            )
    payload = {
        "schema_version": registry.schema_version,
        "source_count": len(registry.resources),
        "processing_source_count": 0 if processing is None else len(processing.sources),
        "sources": [
            {
                "id": source.id,
                "version": source.version,
                "access_class": source.access_class.value,
                "connector": source.connector,
                "parser": source.parser,
            }
            for source in registry.resources
        ],
    }
    if processing is not None:
        payload["processing"] = [
            {
                "source_id": record.source_id,
                "state": record.state.value,
                "promotion_boundary": record.promotion_boundary.value,
                "dossier": record.dossier,
            }
            for record in processing.sources
        ]
    _print_json(payload)
    return 0


def materialize_artifact(args: argparse.Namespace) -> int:
    """Restore one external CAS object without persisting its backend location."""

    store = artifact_store_from_uri(args.store)
    materialized = materialize_stored_object(
        store,
        args.sha256,
        args.output,
        expected_byte_size=args.expected_byte_size,
    )
    _print_json(
        {
            "byte_size": materialized.byte_size,
            "output": str(Path(args.output)),
            "sha256": materialized.sha256,
            "uri": materialized.uri,
        }
    )
    return 0


def lock_mining_release(args: argparse.Namespace) -> int:
    """Create a machine-portable byte lock for one materialized release spec."""

    manifest = build_mining_release_lock(args.spec, args.output)
    _print_json(
        {
            "artifact_count": len(manifest["artifacts"]),
            "output": str(Path(args.output)),
            "release_fingerprint": manifest["release_fingerprint"],
            "release_id": manifest["release_id"],
        }
    )
    return 0


def verify_mining_release(args: argparse.Namespace) -> int:
    """Verify the same release bytes under a caller-selected local root."""

    store = artifact_store_from_uri(args.store) if args.store else None
    report = verify_mining_release_lock(
        args.manifest,
        release_root=args.root,
        require_optional=args.require_optional,
        artifact_store=store,
        require_cas_objects=args.require_cas_objects,
        verify_cas_content=args.verify_cas_content,
    )
    _print_json(report)
    return 0 if report["valid"] else 1


def sync_registered_source(args: argparse.Namespace) -> int:
    registry = load_source_registry(args.registry)
    source = registry.by_id(args.source_id)
    parameters = _load_mapping(args.parameters) if args.parameters else {}
    policy_gate = SourcePolicyGate(registry)
    storage_decision = policy_gate.artifact_storage(
        source,
        store_uri=args.store,
        encrypted_at_rest=args.encrypted_at_rest,
    )
    if not storage_decision.allowed:
        raise PermissionError(
            f"Storage policy rejected {source.id}: {', '.join(storage_decision.reasons)}"
        )
    store = artifact_store_from_uri(args.store)
    artifacts = sync_source(
        source=source,
        request=SourceRequest(
            source_id=args.source_id,
            source_version=args.source_version,
            parameters=parameters,
        ),
        store=store,
        policy_gate=policy_gate,
        checkpoint_path=args.output,
    )
    _print_json({"artifact_count": len(artifacts), "output": args.output})
    return 0


def build_dataset(args: argparse.Namespace) -> int:
    registry = load_source_registry(args.registry)
    source = registry.by_id(args.source_id)
    artifacts = load_source_artifacts(args.artifacts)
    wrong_sources = sorted(
        {artifact.source_id for artifact in artifacts if artifact.source_id != source.id}
    )
    if wrong_sources:
        raise ValueError(f"Artifact manifest contains other sources: {', '.join(wrong_sources)}")
    manifest = materialize_documents(
        source=source,
        artifacts=artifacts,
        store=artifact_store_from_uri(args.store),
        output_path=args.output,
    )
    _print_json({"document_count": manifest.document_count, "output": args.output})
    return 0


def inspect_dataset(args: argparse.Namespace) -> int:
    """Write a reusable source profile and optionally gate structural issues."""

    documents = load_documents(args.documents)
    annotations = () if args.annotations is None else load_annotations(args.annotations)
    profile = build_dataset_profile(documents, annotations)
    blocking_issue_count = profile_blocking_issue_count(profile)
    write_json(args.output, profile)
    _print_json(
        {
            "annotation_count": len(annotations),
            "blocking_issue_count": blocking_issue_count,
            "document_count": len(documents),
            "output": args.output,
        }
    )
    return 1 if args.strict and blocking_issue_count else 0


def attach_dataset_block_evidence(args: argparse.Namespace) -> int:
    """Enrich proposals with source-section provenance and evidence tiers."""

    result = attach_block_evidence(
        load_documents(args.documents),
        load_annotations(args.annotations),
        load_block_evidence_policy(args.policy),
    )
    annotations_sha256 = write_jsonl(
        args.output,
        (annotation.to_dict() for annotation in result.annotations),
    )
    report = {
        **result.report,
        "inputs": {
            "documents": str(Path(args.documents)),
            "documents_sha256": sha256_file(args.documents),
            "annotations": str(Path(args.annotations)),
            "annotations_sha256": sha256_file(args.annotations),
            "policy": str(Path(args.policy)),
            "policy_sha256": sha256_file(args.policy),
        },
        "outputs": {
            "annotations": str(Path(args.output)),
            "annotations_sha256": annotations_sha256,
        },
    }
    write_json(args.report_output, report)
    _print_json(
        {
            "annotation_count": len(result.annotations),
            "annotation_tier_counts": result.report["annotation_tier_counts"],
            "output": args.output,
            "report": args.report_output,
        }
    )
    return 0


def reconcile_duplicates(args: argparse.Namespace) -> int:
    """Materialize exact-text consensus and a separate disagreement queue."""

    result = reconcile_exact_duplicates(
        load_documents(args.documents),
        load_annotations(args.annotations),
        labeler_id=args.labeler_id,
    )
    write_jsonl(
        args.documents_output,
        (document.to_dict() for document in result.documents),
    )
    write_jsonl(
        args.annotations_output,
        (annotation.to_dict() for annotation in result.training_annotations),
    )
    write_jsonl(
        args.review_output,
        (annotation.to_dict() for annotation in result.review_annotations),
    )
    write_jsonl(
        args.mapping_output,
        (mapping.to_dict() for mapping in result.document_mappings),
    )
    write_json(args.report_output, result.report.to_dict())
    _print_json(
        {
            "document_count": len(result.documents),
            "training_annotation_count": len(result.training_annotations),
            "review_annotation_count": len(result.review_annotations),
            "exact_micro_jaccard": result.report.exact_micro_jaccard,
            "report": args.report_output,
        }
    )
    return 0


def fuse_datasets(args: argparse.Namespace) -> int:
    """Run deterministic cross-source fusion from a strict plan."""

    result = run_corpus_fusion_plan(args.plan)
    _print_json(result.model_dump(mode="json"))
    return 0


def harmonize_dataset(args: argparse.Namespace) -> int:
    """Apply source-aware label alignment with terminology existence checks."""

    repository = SQLiteTerminologyRepository(
        args.index,
        expected_source_paths=tuple(args.source),
        expected_alias_overlay_paths=tuple(args.alias_overlay_source),
    )
    try:
        result = harmonize_annotations(
            load_documents(args.documents),
            load_annotations(args.annotations),
            repository,
            load_annotation_harmonization_policy(args.policy),
        )
    finally:
        repository.close()
    annotations_sha256 = write_jsonl(
        args.output, (annotation.to_dict() for annotation in result.annotations)
    )
    decisions_sha256 = write_jsonl(args.decisions_output, result.decisions)
    report = {
        **result.report,
        "inputs": {
            "documents": str(Path(args.documents)),
            "documents_sha256": sha256_file(args.documents),
            "annotations": str(Path(args.annotations)),
            "annotations_sha256": sha256_file(args.annotations),
            "policy": str(Path(args.policy)),
            "policy_sha256": sha256_file(args.policy),
            "terminology_index": str(Path(args.index)),
            "terminology_input_fingerprint": repository.metadata.get("input_fingerprint", ""),
        },
        "outputs": {
            "annotations": str(Path(args.output)),
            "annotations_sha256": annotations_sha256,
            "decisions": str(Path(args.decisions_output)),
            "decisions_sha256": decisions_sha256,
        },
    }
    write_json(args.report_output, report)
    _print_json(
        {
            "annotation_count": len(result.annotations),
            "decision_counts": result.report["decision_counts"],
            "output": args.output,
            "report": args.report_output,
        }
    )
    return 0


def curate_annotation_dataset(args: argparse.Namespace) -> int:
    """Materialize a policy-specific view without mutating source proposals."""

    result = curate_annotations(
        load_annotations(args.annotations),
        load_annotation_curation_policy(args.policy),
    )
    write_jsonl(
        args.accepted_output,
        (annotation.to_dict() for annotation in result.accepted),
    )
    write_jsonl(
        args.rejected_output,
        (annotation.to_dict() for annotation in result.rejected),
    )
    write_json(args.report_output, result.report)
    _print_json(
        {
            "accepted_count": len(result.accepted),
            "rejected_count": len(result.rejected),
            "report": args.report_output,
        }
    )
    return 0


def export_span_training_dataset(args: argparse.Namespace) -> int:
    """Compile raw character spans for tokenizer-specific training downstream."""

    documents = load_documents(args.documents)
    annotations = load_annotations(args.annotations)
    manifest = export_span_dataset(
        documents,
        annotations,
        load_dataset_splits(args.split_manifest),
        SpanDatasetConfig(
            max_characters=args.max_characters,
            entity_types=tuple(args.entity_type),
            include_empty_chunks=not args.drop_empty_chunks,
            empty_chunk_rate=args.empty_chunk_rate,
        ),
        output_path=args.output,
        manifest_path=args.manifest_output,
        documents_path=args.documents,
        annotations_path=args.annotations,
        split_manifest_path=args.split_manifest,
    )
    _print_json(manifest)
    return 0


def build_exact_quote_curriculum_dataset(args: argparse.Namespace) -> int:
    """Compile licensed source spans into a train-only exact-quote curriculum."""

    report = build_exact_quote_curriculum(
        ExactQuoteCurriculumConfig(
            source_id=args.source_id,
            source_registry_path=Path(args.registry),
            spans_path=Path(args.spans),
            spans_manifest_path=Path(args.spans_manifest),
            output_dir=Path(args.output_dir),
            allowed_labels=tuple(sorted(set(args.label))),
        )
    )
    _print_json(report)
    return 0


def build_lexicon(args: argparse.Namespace) -> int:
    """Write mined mention hypotheses and a separate ambiguity report."""

    documents, annotations = _load_selected_mined_records(
        documents_path=args.documents,
        annotations_path=args.annotations,
        split_manifest=args.split_manifest,
        split=args.split,
    )
    result = build_mention_inventory(
        documents,
        annotations,
        min_occurrences=args.min_occurrences,
        min_documents=args.min_documents,
    )
    inventory_sha256 = write_jsonl(
        args.output,
        (entry.to_dict() for entry in result.entries),
    )
    conflicts_sha256 = write_jsonl(args.conflicts_output, result.conflicts)
    report = {
        **result.report,
        "inputs": {
            "documents": str(Path(args.documents)),
            "documents_sha256": sha256_file(args.documents),
            "annotations": str(Path(args.annotations)),
            "annotations_sha256": sha256_file(args.annotations),
        },
        "selection": {
            "split_manifest": (
                None if args.split_manifest is None else str(Path(args.split_manifest))
            ),
            "split_manifest_sha256": (
                None if args.split_manifest is None else sha256_file(args.split_manifest)
            ),
            "split": args.split,
        },
        "outputs": {
            "inventory": str(Path(args.output)),
            "inventory_sha256": inventory_sha256,
            "conflicts": str(Path(args.conflicts_output)),
            "conflicts_sha256": conflicts_sha256,
        },
    }
    write_json(args.report_output, report)
    _print_json(
        {
            "ambiguous_mention_count": len(result.conflicts),
            "entry_count": len(result.entries),
            "output": args.output,
            "report": args.report_output,
        }
    )
    return 0


def crosswalk_lexicon(args: argparse.Namespace) -> int:
    """Write terminology review proposals while rejecting stale derived indexes."""

    repository = SQLiteTerminologyRepository(
        args.index,
        expected_source_paths=tuple(args.source),
        expected_alias_overlay_paths=tuple(args.alias_overlay_source),
    )
    try:
        result = crosswalk_mentions(
            load_mention_inventory(args.inventory),
            repository,
            load_crosswalk_policies(args.policy),
            terminology_metadata=repository.metadata,
            workers=args.workers,
            query_limit=args.query_limit,
            candidate_output_limit=args.candidate_output_limit,
            lexical_fallback=args.lexical_fallback,
        )
    finally:
        repository.close()
    write_jsonl(args.output, (record.to_dict() for record in result.records))
    write_json(args.report_output, result.report)
    _print_json(
        {
            "entry_count": len(result.records),
            "output": args.output,
            "report": args.report_output,
            "status_entry_counts": result.report["status_entry_counts"],
            "unique_exact_entry_count": result.report["unique_exact_entry_count"],
        }
    )
    return 0


def propose_dailymed_product_aliases(args: argparse.Namespace) -> int:
    """Build train or held-out product alias hypotheses from exact identity links."""

    links_sha256 = sha256_file(args.links)
    split_manifest_sha256 = sha256_file(args.split_manifest)
    result = build_dailymed_product_alias_proposals(
        tuple(_load_jsonl_mappings(args.links)),
        selected_document_ids=load_split_document_ids(args.split_manifest, args.split),
        split_name=args.split,
        links_sha256=links_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )
    proposals_sha256 = write_jsonl(args.output, result.proposals)
    decisions_sha256 = write_jsonl(args.decisions_output, result.decisions)
    report = {
        **result.report,
        "inputs": {
            "links": str(Path(args.links)),
            "links_sha256": links_sha256,
            "split_manifest": str(Path(args.split_manifest)),
            "split_manifest_sha256": split_manifest_sha256,
        },
        "outputs": {
            "proposals": str(Path(args.output)),
            "proposals_sha256": proposals_sha256,
            "decisions": str(Path(args.decisions_output)),
            "decisions_sha256": decisions_sha256,
        },
    }
    write_json(args.report_output, report)
    _print_json(
        {
            "split": args.split,
            "proposal_count": len(result.proposals),
            "ambiguous_target_alias_count": result.report["ambiguous_target_alias_count"],
            "output": args.output,
            "report": args.report_output,
        }
    )
    return 0


def attach_exact_crosswalk_links(args: argparse.Namespace) -> int:
    """Materialize exact crosswalk evidence without changing source spans."""

    result = materialize_exact_crosswalk_links(
        load_annotations(args.annotations),
        load_crosswalk_rows(args.crosswalk),
        load_crosswalk_link_policy(args.policy),
    )
    output_sha256 = write_jsonl(
        args.output, (annotation.to_dict() for annotation in result.annotations)
    )
    report = {
        **result.report,
        "inputs": {
            "annotations": str(Path(args.annotations)),
            "annotations_sha256": sha256_file(args.annotations),
            "crosswalk": str(Path(args.crosswalk)),
            "crosswalk_sha256": sha256_file(args.crosswalk),
            "policy": str(Path(args.policy)),
            "policy_sha256": sha256_file(args.policy),
        },
        "output": str(Path(args.output)),
        "output_sha256": output_sha256,
    }
    write_json(args.report_output, report)
    _print_json(
        {
            "annotation_decision_counts": report["annotation_decision_counts"],
            "output": args.output,
            "report": args.report_output,
        }
    )
    return 0


def propose_linked_aliases(args: argparse.Namespace) -> int:
    """Build source-pinned alias proposals from human concept-linked spans."""

    result = build_linked_alias_proposals(
        load_documents(args.documents),
        load_annotations(args.annotations),
        load_source_artifacts(args.artifacts),
        load_linked_alias_policy(args.policy),
    )
    proposals_sha256 = write_jsonl(args.output, result.proposals)
    decisions_sha256 = write_jsonl(args.decisions_output, result.decisions)
    report = {
        **result.report,
        "inputs": {
            "documents": str(Path(args.documents)),
            "documents_sha256": sha256_file(args.documents),
            "annotations": str(Path(args.annotations)),
            "annotations_sha256": sha256_file(args.annotations),
            "artifacts": str(Path(args.artifacts)),
            "artifacts_sha256": sha256_file(args.artifacts),
            "policy": str(Path(args.policy)),
        },
        "outputs": {
            "proposals": str(Path(args.output)),
            "proposals_sha256": proposals_sha256,
            "decisions": str(Path(args.decisions_output)),
            "decisions_sha256": decisions_sha256,
        },
    }
    write_json(args.report_output, report)
    _print_json(
        {
            "proposal_count": len(result.proposals),
            "proposal_occurrence_count": result.report["proposal_occurrence_count"],
            "decision_counts": result.report["decision_counts"],
            "output": args.output,
            "report": args.report_output,
        }
    )
    return 0


def compile_dailymed_rxnorm(args: argparse.Namespace) -> int:
    """Compile one checksum-pinned official DailyMed mapping artifact."""

    artifacts = load_source_artifacts(args.artifacts)
    if len(artifacts) != 1:
        raise ValueError("DailyMed RxNorm compilation requires exactly one artifact")
    artifact = artifacts[0]
    store = artifact_store_from_uri(args.store)
    with store.open(artifact.object.sha256) as stream:
        report = compile_dailymed_rxnorm_mapping(
            artifact,
            stream,
            output_path=args.output,
            index_path=args.index_output,
            report_path=args.report_output,
        )
    _print_json(report)
    return 0


def audit_dailymed_rxnorm(args: argparse.Namespace) -> int:
    """Emit review-only aliases after validating the terminology source fingerprint."""

    # INVARIANT: validate the derived terminology index against canonical JSONL
    # before querying its internal tables for code and alias membership.
    repository = SQLiteTerminologyRepository(
        args.terminology_index,
        expected_source_paths=tuple(args.source),
    )
    repository.close()
    report = audit_dailymed_rxnorm_mapping(
        args.index,
        args.terminology_index,
        proposals_path=args.proposals_output,
        report_path=args.report_output,
    )
    _print_json(report)
    return 0


def compile_rxnorm_ndc(args: argparse.Namespace) -> int:
    """Compile package-level NDC evidence from one immutable RxNorm archive."""

    report = compile_rxnorm_ndc_index(
        args.source,
        source_version=args.source_version,
        expected_source_sha256=args.expected_source_sha256,
        archive_member=args.archive_member,
        output_path=args.output,
        index_path=args.index_output,
        report_path=args.report_output,
    )
    _print_json(report)
    return 0


def mine_abbreviation_knowledge(args: argparse.Namespace) -> int:
    """Mine split-safe definitions and benchmark their retrieval contribution."""

    if bool(args.index) != bool(args.source):
        raise ValueError("--index and at least one --source must be provided together")
    policy = load_abbreviation_mining_policy(args.policy)
    base_rows = tuple(row for path in args.base_abbreviations for row in _load_jsonl_mappings(path))
    result = mine_abbreviations(
        iter_documents(args.documents),
        load_source_artifacts(args.artifacts),
        load_snapshot_splits(args.split_manifest),
        policy,
        base_abbreviation_rows=base_rows,
    )
    definitions_sha256 = write_jsonl(args.definitions_output, result.definitions)
    candidates_sha256 = write_jsonl(args.candidates_output, result.candidates)
    table_sha256 = write_jsonl(args.table_output, result.abbreviation_table)
    runtime_table = build_runtime_abbreviation_table(base_rows, result.abbreviation_table)
    runtime_table_sha256 = write_jsonl(args.runtime_table_output, runtime_table)
    conflicts_sha256 = write_jsonl(args.conflicts_output, result.conflicts)

    repository = None
    terminology_metadata: dict[str, str] = {}
    if args.index:
        repository = SQLiteTerminologyRepository(
            args.index,
            expected_source_paths=tuple(args.source),
            expected_alias_overlay_paths=tuple(args.alias_overlay),
        )
        terminology_metadata = dict(repository.metadata)
    try:
        benchmark = benchmark_abbreviation_knowledge(
            result.definitions,
            base_rows,
            result.abbreviation_table,
            evaluation_splits=policy.evaluation_splits,
            repository=repository,
            retrieval_limit=args.retrieval_limit,
        )
    finally:
        if repository is not None:
            repository.close()
    benchmark_sha256 = write_json(args.benchmark_output, benchmark)

    # Rebuild evidence uses content fingerprints rather than machine-specific absolute paths.
    report = {
        **result.report,
        "runtime_abbreviation_table_count": len(runtime_table),
        "inputs": {
            "documents": str(Path(args.documents)),
            "documents_sha256": sha256_file(args.documents),
            "artifacts": str(Path(args.artifacts)),
            "artifacts_sha256": sha256_file(args.artifacts),
            "split_manifest": str(Path(args.split_manifest)),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "policy": str(Path(args.policy)),
            "policy_sha256": sha256_file(args.policy),
            "base_abbreviations": [str(Path(path)) for path in args.base_abbreviations],
            "base_abbreviation_sha256": [sha256_file(path) for path in args.base_abbreviations],
            "terminology_index": None if not args.index else str(Path(args.index)),
            "terminology_input_fingerprint": terminology_metadata.get("input_fingerprint", ""),
        },
        "outputs": {
            "definitions": str(Path(args.definitions_output)),
            "definitions_sha256": definitions_sha256,
            "candidates": str(Path(args.candidates_output)),
            "candidates_sha256": candidates_sha256,
            "abbreviation_table": str(Path(args.table_output)),
            "abbreviation_table_sha256": table_sha256,
            "runtime_abbreviation_table": str(Path(args.runtime_table_output)),
            "runtime_abbreviation_table_sha256": runtime_table_sha256,
            "conflicts": str(Path(args.conflicts_output)),
            "conflicts_sha256": conflicts_sha256,
            "benchmark": str(Path(args.benchmark_output)),
            "benchmark_sha256": benchmark_sha256,
        },
    }
    write_json(args.report_output, report)
    retrieval_summary = benchmark.get("retrieval", {})
    _print_json(
        {
            "definition_count": report["definition_count"],
            "candidate_count": report["candidate_count"],
            "abbreviation_table_count": report["abbreviation_table_count"],
            "runtime_abbreviation_table_count": len(runtime_table),
            "conflict_count": report["conflict_count"],
            "benchmark": {
                "baseline": benchmark["baseline"],
                "enriched": benchmark["enriched"],
                "unique_pairs": benchmark["unique_pairs"],
                "retrieval": {
                    key: value
                    for key, value in retrieval_summary.items()
                    if key != "cases"
                },
            },
            "report": args.report_output,
        }
    )
    return 0


def link_dailymed_products(args: argparse.Namespace) -> int:
    """Join two official source identities and withhold every ambiguous product."""

    report = link_dailymed_products_to_rxnorm(
        args.documents,
        dailymed_mapping_index_path=args.dailymed_mapping_index,
        rxnorm_ndc_index_path=args.rxnorm_ndc_index,
        links_path=args.links_output,
        decisions_path=args.decisions_output,
        report_path=args.report_output,
    )
    _print_json(report)
    return 0


def compile_alias_knowledge(args: argparse.Namespace) -> int:
    """Compile source-pinned aliases and a compact recognition dictionary."""

    repository = SQLiteTerminologyRepository(
        args.index,
        expected_source_paths=tuple(args.source),
        expected_alias_overlay_paths=tuple(args.base_alias_overlay),
    )
    try:
        result = compile_mined_aliases(
            (
                row
                for proposal_path in args.proposals
                for row in _load_jsonl_mappings(proposal_path)
            ),
            repository,
            load_alias_promotion_policy(args.policy),
        )
    finally:
        repository.close()
    overlay_sha256 = write_jsonl(args.overlay_output, result.alias_overlays)
    recognition_sha256 = write_jsonl(
        args.recognition_output,
        result.recognition_concepts,
    )
    decisions_sha256 = write_jsonl(args.decisions_output, result.decisions)
    report = {
        **result.report,
        "inputs": {
            "proposals": [str(Path(path)) for path in args.proposals],
            "terminology_index": str(Path(args.index)),
            "terminology_input_fingerprint": repository.metadata.get(
                "input_fingerprint",
                "",
            ),
        },
        "outputs": {
            "alias_overlay": str(Path(args.overlay_output)),
            "alias_overlay_sha256": overlay_sha256,
            "recognition_dictionary": str(Path(args.recognition_output)),
            "recognition_dictionary_sha256": recognition_sha256,
            "decisions": str(Path(args.decisions_output)),
            "decisions_sha256": decisions_sha256,
        },
    }
    write_json(args.report_output, report)
    _print_json(
        {
            "overlay_alias_count": report["overlay_alias_count"],
            "recognition_concept_count": report["recognition_concept_count"],
            "decision_counts": report["decision_counts"],
            "report": args.report_output,
        }
    )
    return 0


def compile_graph_knowledge(args: argparse.Namespace) -> int:
    """Deduplicate terminology, mined entities, relations, and source evidence."""

    documents = () if args.documents is None else load_documents(args.documents)
    annotations = () if args.annotations is None else load_annotations(args.annotations)
    relations = () if args.relations is None else load_relations(args.relations)
    defaults = GraphCompilationConfig()
    report = compile_knowledge_graph(
        terminology_paths=tuple(args.terminology_source),
        alias_overlay_paths=tuple(args.alias_overlay),
        documents=documents,
        annotations=annotations,
        relations=relations,
        config=GraphCompilationConfig(
            accepted_layers=(
                tuple(AnnotationLayer(value) for value in args.accepted_layer)
                or defaults.accepted_layers
            ),
            accepted_review_statuses=(
                tuple(ReviewStatus(value) for value in args.accepted_review_status)
                or defaults.accepted_review_statuses
            ),
            include_entity_types=tuple(args.entity_type),
            include_unlinked_terms=not args.linked_only,
            include_structured_terminology_relations=(not args.no_structured_terminology_relations),
            relation_endpoints_only=args.relation_endpoints_only,
            require_canonical_concepts=args.canonical_concepts_only,
            preferred_code_systems_by_entity_type=(
                _parse_preferred_code_systems(args.preferred_code_system)
            ),
        ),
        nodes_output=args.nodes_output,
        edges_output=args.edges_output,
        evidence_output=args.evidence_output,
        report_output=args.report_output,
        documents_path=args.documents,
        annotations_path=args.annotations,
        relations_path=args.relations,
    )
    _print_json(report)
    return 0


def _parse_preferred_code_systems(
    values: list[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Parse typed graph endpoint selectors and reject malformed CLI values."""

    grouped: dict[str, list[str]] = {}
    for raw in values:
        entity_text, separator, system_text = raw.partition("=")
        if not separator:
            raise ValueError("--preferred-code-system must use ENTITY_TYPE=CODE_SYSTEM")
        entity_type = EntityType(entity_text.strip()).value
        code_system = CodeSystem(system_text.strip()).value
        systems = grouped.setdefault(entity_type, [])
        if code_system in systems:
            raise ValueError(f"Duplicate preferred code system {code_system!r} for {entity_type!r}")
        systems.append(code_system)
    return tuple((entity_type, tuple(systems)) for entity_type, systems in sorted(grouped.items()))


def compile_obo_ontology(args: argparse.Namespace) -> int:
    """Compile a source namespace without promoting its labels into runtime NER."""

    report = compile_obo_graph_release(
        input_path=args.input,
        output_dir=args.output_dir,
        config=OBOGraphCompilationConfig(
            source_id=args.source_id,
            source_version=args.source_version,
            iri_prefix=args.iri_prefix,
            code_system=CodeSystem(args.code_system),
            entity_type=EntityType(args.entity_type),
        ),
    )
    _print_json(report)
    return 0


def compile_hpo_association_knowledge(args: argparse.Namespace) -> int:
    """Compile full HPOA evidence while keeping negated phenotypes separate."""

    report = compile_hpo_associations(
        hpoa_path=args.hpoa,
        genes_path=args.genes,
        hpo_concepts_path=args.hpo_concepts,
        output_dir=args.output_dir,
        source_version=args.source_version,
    )
    _print_json(report)
    return 0


def compile_recognition_knowledge_artifact(args: argparse.Namespace) -> int:
    """Compile corpus-backed, code-free recognition concepts for NER evaluation."""

    inventory_sha256 = sha256_file(args.inventory)
    baseline_entries = tuple(
        entry
        for dictionary_path in args.baseline_dictionary
        for entry in DictionaryStore.load_entries_jsonl(dictionary_path)
    )
    result = compile_recognition_knowledge(
        load_mention_inventory(args.inventory),
        load_recognition_knowledge_policy(args.policy),
        inventory_sha256=inventory_sha256,
        baseline_entries=baseline_entries,
    )
    output_sha256 = write_jsonl(args.output, result.concepts)
    decisions_sha256 = write_jsonl(args.decisions_output, result.decisions)
    report = {
        **result.report,
        "inputs": {
            "inventory": str(Path(args.inventory)),
            "inventory_sha256": inventory_sha256,
            "policy": str(Path(args.policy)),
            "baseline_dictionaries": [str(Path(path)) for path in args.baseline_dictionary],
        },
        "outputs": {
            "recognition_dictionary": str(Path(args.output)),
            "recognition_dictionary_sha256": output_sha256,
            "decisions": str(Path(args.decisions_output)),
            "decisions_sha256": decisions_sha256,
        },
    }
    write_json(args.report_output, report)
    _print_json(
        {
            "recognition_concept_count": len(result.concepts),
            "decision_counts": result.report["decision_counts"],
            "output": args.output,
            "report": args.report_output,
        }
    )
    return 0


def benchmark_recognition_knowledge(args: argparse.Namespace) -> int:
    """Measure exact span/type changes before promoting mined NER knowledge."""

    documents, annotations = _load_selected_mined_records(
        documents_path=args.documents,
        annotations_path=args.annotations,
        split_manifest=args.split_manifest,
        split=args.split,
    )
    required_metadata = tuple(args.require_document_metadata)
    if required_metadata:
        selection = select_mined_records_with_metadata(
            documents,
            annotations,
            required_metadata,
        )
        documents, annotations = selection.documents, selection.annotations
    report = benchmark_recognition_dictionary(
        documents,
        annotations,
        DictionaryStore.from_jsonl(args.baseline_dictionary),
        DictionaryStore.from_jsonl(args.additional_dictionary),
        entity_types=tuple(EntityType(value) for value in args.entity_type),
    )
    report["selection"] = {
        "split_manifest": args.split_manifest,
        "split": args.split,
        "required_document_metadata": list(required_metadata),
    }
    write_json(args.output, report)
    _print_json(
        {
            "baseline": report["baseline"]["metrics"],
            "enriched": report["enriched"]["metrics"],
            "delta": report["delta"],
            "output": args.output,
        }
    )
    return 0


def propose_labels(args: argparse.Namespace) -> int:
    """Stream document batches through a labeler and external-sort its output."""

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    config = _load_mapping(args.adapter_config) if args.adapter_config else None
    labeler = _load_labeler(args.adapter, config)
    if args.hosted:
        labeler = PolicyAwareProposalLabelerAdapter(
            labeler,
            allow_document=lambda document: document.hosted_processing_allowed,
        )
    result = materialize_annotation_manifest(
        args.output,
        _iter_validated_proposals(
            args.documents,
            labeler,
            batch_size=args.batch_size,
        ),
    )
    _print_json(
        {
            "proposal_count": result.annotation_count,
            "duplicate_count": result.duplicate_count,
            "sha256": result.sha256,
            "output": args.output,
        }
    )
    return 0


def _iter_validated_proposals(
    documents_path: str | Path,
    labeler: ProposalLabelerPort,
    *,
    batch_size: int,
) -> Iterator[AnnotationProposal]:
    """Validate proposal ownership and raw offsets one bounded batch at a time."""

    batch: list[MinedDocument] = []
    for document in iter_documents(documents_path):
        batch.append(document)
        if len(batch) == batch_size:
            yield from _validated_batch_proposals(labeler, batch)
            batch = []
    if batch:
        yield from _validated_batch_proposals(labeler, batch)


def _validated_batch_proposals(
    labeler: ProposalLabelerPort,
    documents: list[MinedDocument],
) -> Iterator[AnnotationProposal]:
    documents_by_id = {document.document_id: document for document in documents}
    for proposal in labeler.propose(documents):
        document = documents_by_id.get(proposal.document_id)
        if document is None:
            raise ValueError(
                f"Proposal references a document outside its input batch: {proposal.document_id!r}"
            )
        proposal.validate_offsets(document)
        yield proposal


def propose_relations(args: argparse.Namespace) -> int:
    """Run a relation adapter and validate every document/endpoint reference."""

    documents = load_documents(args.documents)
    annotations = load_annotations(args.annotations)
    config = _load_mapping(args.adapter_config) if args.adapter_config else None
    labeler = _load_relation_labeler(args.adapter, config)
    relations = tuple(labeler.propose(documents, annotations))
    documents_by_id = {document.document_id: document for document in documents}
    annotations_by_id = {annotation.annotation_id: annotation for annotation in annotations}
    for relation in relations:
        document = documents_by_id.get(relation.document_id)
        if document is None:
            raise ValueError(f"Relation references unknown document {relation.document_id!r}")
        head = annotations_by_id.get(relation.head_annotation_id)
        tail = annotations_by_id.get(relation.tail_annotation_id)
        if head is None or tail is None:
            raise ValueError(f"Relation {relation.relation_id!r} has an unknown endpoint")
        if head.document_id != relation.document_id or tail.document_id != relation.document_id:
            raise ValueError(f"Relation {relation.relation_id!r} crosses documents")
        if relation.evidence_span is not None:
            start, end = relation.evidence_span
            if start < 0 or end <= start or end > len(document.text):
                raise ValueError(f"Relation {relation.relation_id!r} has invalid evidence span")
    ordered = sorted(relations, key=lambda item: item.relation_id)
    write_jsonl(args.output, (relation.to_dict() for relation in ordered))
    _print_json({"relation_count": len(ordered), "output": args.output})
    return 0


def mine_cooccurrence(args: argparse.Namespace) -> int:
    """Mine auditable sentence co-occurrence from a source-pinned training slice."""

    documents = load_documents(args.documents)
    annotations = load_annotations(args.annotations)
    if (args.split_manifest is None) != (args.split is None):
        raise ValueError("--split-manifest and --split must be provided together")
    selected_document_ids = (
        None
        if args.split_manifest is None
        else load_split_document_ids(args.split_manifest, args.split)
    )
    result = mine_cooccurrence_relations(
        documents,
        annotations,
        load_cooccurrence_policy(args.policy),
        selected_document_ids=selected_document_ids,
    )
    relations_sha256 = write_jsonl(
        args.output,
        (relation.to_dict() for relation in result.relations),
    )
    report = {
        **result.report,
        "inputs": {
            "documents": str(Path(args.documents)),
            "documents_sha256": sha256_file(args.documents),
            "annotations": str(Path(args.annotations)),
            "annotations_sha256": sha256_file(args.annotations),
            "policy": str(Path(args.policy)),
            "policy_sha256": sha256_file(args.policy),
            "split_manifest": (
                None if args.split_manifest is None else str(Path(args.split_manifest))
            ),
            "split_manifest_sha256": (
                None if args.split_manifest is None else sha256_file(args.split_manifest)
            ),
            "split": args.split,
        },
        "outputs": {
            "relations": str(Path(args.output)),
            "relations_sha256": relations_sha256,
        },
    }
    write_json(args.report_output, report)
    _print_json(
        {
            "relation_count": len(result.relations),
            "supported_semantic_pair_count": result.report["counters"]["supported_semantic_pairs"],
            "output": args.output,
            "report": args.report_output,
        }
    )
    return 0


def export_review(args: argparse.Namespace) -> int:
    payload = JsonlReviewBackend().export(
        load_documents(args.documents),
        load_annotations(args.proposals),
    )
    fingerprint = write_text(args.output, payload)
    _print_json({"output": args.output, "sha256": fingerprint})
    return 0


def import_review(args: argparse.Namespace) -> int:
    payload = Path(args.input).read_text(encoding="utf-8")
    proposals = JsonlReviewBackend().import_reviewed(payload)
    write_jsonl(args.output, (proposal.to_dict() for proposal in proposals))
    _print_json({"proposal_count": len(proposals), "output": args.output})
    return 0


def review_quality(args: argparse.Namespace) -> int:
    documents = load_documents(args.documents)
    proposals = load_annotations(args.proposals)
    relations = () if args.relations is None else load_relations(args.relations)
    report = ReviewAgreementEvaluator().evaluate(documents, proposals, relations)
    issues = GoldAgreementGate().validate(
        report,
        has_gold_relations=any(
            relation.layer.value in {"gold", "challenge"} for relation in relations
        ),
    )
    payload = {**report.to_dict(), "blocking_issues": list(issues)}
    write_json(args.output, payload)
    _print_json({"blocking_issue_count": len(issues), "output": args.output})
    return 1 if issues else 0


def report_coverage(args: argparse.Namespace) -> int:
    documents = load_documents(args.documents)
    proposals = load_annotations(args.proposals)
    target_payload = _load_mapping(args.targets)
    raw_targets = target_payload.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError("Coverage config requires a targets list")
    targets: list[CoverageTarget] = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, Mapping):
            raise ValueError("Each coverage target must be an object")
        dimensions = raw_target.get("dimensions")
        if not isinstance(dimensions, Mapping):
            raise ValueError("Coverage target dimensions must be an object")
        targets.append(
            CoverageTarget(
                tuple(sorted((str(key), str(value)) for key, value in dimensions.items())),
                target=int(raw_target["target"]),
            )
        )
    planner = CoverageCubePlanner(targets)
    report = planner.report(args.snapshot_id, documents, proposals)
    priorities = planner.priorities(documents, proposals)
    payload = {
        "snapshot_id": report.snapshot_id,
        "mean_gap": report.mean_gap,
        "cells": [
            {
                "dimensions": dict(cell.dimensions),
                "observed": cell.observed,
                "target": cell.target,
                "human_reviewed": cell.human_reviewed,
                "synthetic": cell.synthetic,
                "gap_ratio": cell.gap_ratio,
            }
            for cell in report.cells
        ],
        "priorities": [asdict(priority) for priority in priorities],
    }
    write_json(args.output, payload)
    _print_json({"cell_count": len(report.cells), "output": args.output})
    return 0


def freeze_snapshot(args: argparse.Namespace) -> int:
    documents = load_documents(args.documents)
    annotations = () if args.annotations is None else load_annotations(args.annotations)
    relations = () if args.relations is None else load_relations(args.relations)
    source_fingerprints = tuple(args.source_fingerprint)
    for fingerprint in source_fingerprints:
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("--source-fingerprint must be a lowercase SHA-256 digest")
    if args.artifacts is not None:
        source_fingerprints += tuple(
            artifact.object.sha256 for artifact in load_source_artifacts(args.artifacts)
        )
    snapshot = SnapshotBuilder(
        split_config=SnapshotSplitConfig(
            development_fraction=args.development_fraction,
            development_sources=frozenset(args.development_source),
            challenge_sources=frozenset(args.challenge_source),
            challenge_templates=frozenset(args.challenge_template),
            hash_salt=args.hash_salt,
            max_synthetic_train_fraction=args.max_synthetic_fraction,
        ),
        # SCALING: source group IDs already isolate DailyMed label families, so
        # callers may skip expensive whole-document SimHash while retaining exact dedup.
        deduplicator=StableTextDeduplicator(include_near=args.dedup_mode == "near"),
        agreement_gate=(None if args.skip_agreement_gate else GoldAgreementGate()),
    ).freeze(
        version=args.version,
        created_at=args.created_at,
        output_dir=args.output_dir,
        documents=documents,
        annotations=annotations,
        relations=relations,
        source_fingerprints=source_fingerprints,
        write_parquet=not args.manifest_only,
    )
    _print_json(snapshot.to_dict())
    return 0


def run_plan(args: argparse.Namespace) -> int:
    result = run_mining_plan(args.plan)
    _print_json(result.model_dump(mode="json"))
    return 0


def _load_labeler(reference: str, config: Mapping[str, Any] | None) -> ProposalLabelerPort:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Adapter must use module:attribute form")
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory):
        raise TypeError(f"Adapter factory {reference!r} is not callable")
    typed_factory = cast(Callable[..., object], factory)
    labeler = typed_factory() if config is None else typed_factory(config)
    if not callable(getattr(labeler, "propose", None)):
        raise TypeError(f"Adapter factory {reference!r} did not return a proposal labeler")
    return cast(ProposalLabelerPort, labeler)


def _load_relation_labeler(
    reference: str,
    config: Mapping[str, Any] | None,
) -> RelationLabelerPort:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Adapter must use module:attribute form")
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory):
        raise TypeError(f"Adapter factory {reference!r} is not callable")
    typed_factory = cast(Callable[..., object], factory)
    labeler = typed_factory() if config is None else typed_factory(config)
    if not callable(getattr(labeler, "propose", None)):
        raise TypeError(f"Adapter factory {reference!r} did not return a relation labeler")
    return cast(RelationLabelerPort, labeler)


def _load_mapping(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected mapping in {path}")
    return {str(key): value for key, value in raw.items()}


def _load_selected_mined_records(
    *,
    documents_path: str | Path,
    annotations_path: str | Path,
    split_manifest: str | Path | None,
    split: str | None,
) -> tuple[tuple[MinedDocument, ...], tuple[AnnotationProposal, ...]]:
    """Load full records, then apply one frozen split without rewriting offsets."""

    documents = load_documents(documents_path)
    annotations = load_annotations(annotations_path)
    if (split_manifest is None) != (split is None):
        raise ValueError("--split-manifest and --split must be provided together")
    if split_manifest is None or split is None:
        return documents, annotations
    selection = select_mined_records(
        documents,
        annotations,
        load_split_document_ids(split_manifest, split),
    )
    return selection.documents, selection.annotations


def _load_jsonl_mappings(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{source}:{line_number}: expected JSON object")
            yield {str(key): value for key, value in raw.items()}


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
