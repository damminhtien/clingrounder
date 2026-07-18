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

from medical_kg_nlp.mining.coverage import CoverageCubePlanner, CoverageTarget
from medical_kg_nlp.mining.crosswalk import crosswalk_mentions, load_crosswalk_policies
from medical_kg_nlp.mining.curation import (
    curate_annotations,
    load_annotation_curation_policy,
)
from medical_kg_nlp.mining.io import (
    load_annotations,
    load_documents,
    load_relations,
    load_source_artifacts,
    write_json,
    write_jsonl,
    write_text,
)
from medical_kg_nlp.mining.labeling import (
    BatchedProposalLabelerAdapter,
    PolicyAwareProposalLabelerAdapter,
)
from medical_kg_nlp.mining.knowledge import (
    compile_mined_aliases,
    load_alias_promotion_policy,
)
from medical_kg_nlp.mining.lexicon import build_mention_inventory, load_mention_inventory
from medical_kg_nlp.mining.mappings.dailymed_rxnorm import (
    audit_dailymed_rxnorm_mapping,
    compile_dailymed_rxnorm_mapping,
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
from medical_kg_nlp.mining.reconciliation import reconcile_exact_duplicates
from medical_kg_nlp.mining.records import AnnotationProposal, MinedDocument, SourceRequest
from medical_kg_nlp.mining.quality import GoldAgreementGate, ReviewAgreementEvaluator
from medical_kg_nlp.mining.registry import load_source_registry
from medical_kg_nlp.mining.review import JsonlReviewBackend
from medical_kg_nlp.mining.runner import (
    artifact_store_from_uri,
    build_documents,
    run_mining_plan,
    sync_source,
)
from medical_kg_nlp.mining.snapshot import SnapshotBuilder, SnapshotSplitConfig
from medical_kg_nlp.mining.splits import (
    load_split_document_ids,
    select_mined_records,
)
from medical_kg_nlp.terminology import SQLiteTerminologyRepository
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.schema.types import EntityType

__all__ = [
    "build_dataset",
    "build_lexicon",
    "benchmark_recognition_knowledge",
    "crosswalk_lexicon",
    "audit_dailymed_rxnorm",
    "compile_dailymed_rxnorm",
    "compile_alias_knowledge",
    "curate_annotation_dataset",
    "export_review",
    "freeze_snapshot",
    "import_review",
    "inspect_dataset",
    "propose_labels",
    "propose_relations",
    "reconcile_duplicates",
    "report_coverage",
    "review_quality",
    "run_plan",
    "sync_registered_source",
    "validate_registry",
]


def validate_registry(args: argparse.Namespace) -> int:
    registry = load_source_registry(args.registry)
    payload = {
        "schema_version": registry.schema_version,
        "source_count": len(registry.resources),
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
    _print_json(payload)
    return 0


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
    documents = build_documents(
        source=source,
        artifacts=artifacts,
        store=artifact_store_from_uri(args.store),
    )
    write_jsonl(args.output, (document.to_dict() for document in documents))
    _print_json({"document_count": len(documents), "output": args.output})
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
    write_jsonl(args.output, (entry.to_dict() for entry in result.entries))
    write_jsonl(args.conflicts_output, result.conflicts)
    write_json(args.report_output, result.report)
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
    """Write exact terminology proposals while rejecting stale derived indexes."""

    repository = SQLiteTerminologyRepository(
        args.index,
        expected_source_paths=tuple(args.source),
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


def benchmark_recognition_knowledge(args: argparse.Namespace) -> int:
    """Measure exact span/type changes before promoting mined NER knowledge."""

    documents, annotations = _load_selected_mined_records(
        documents_path=args.documents,
        annotations_path=args.annotations,
        split_manifest=args.split_manifest,
        split=args.split,
    )
    report = benchmark_recognition_dictionary(
        documents,
        annotations,
        DictionaryStore.from_jsonl(args.baseline_dictionary),
        DictionaryStore.from_jsonl(args.additional_dictionary),
        entity_types=tuple(EntityType(value) for value in args.entity_type),
    )
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
    documents = load_documents(args.documents)
    config = _load_mapping(args.adapter_config) if args.adapter_config else None
    labeler = _load_labeler(args.adapter, config)
    if args.hosted:
        labeler = PolicyAwareProposalLabelerAdapter(
            labeler,
            allow_document=lambda document: document.hosted_processing_allowed,
        )
    batched = BatchedProposalLabelerAdapter(labeler, batch_size=args.batch_size)
    proposals = tuple(batched.propose(documents))
    documents_by_id = {document.document_id: document for document in documents}
    for proposal in proposals:
        document = documents_by_id.get(proposal.document_id)
        if document is None:
            raise ValueError(f"Proposal references unknown document {proposal.document_id!r}")
        proposal.validate_offsets(document)
    ordered = sorted(proposals, key=lambda item: item.annotation_id)
    write_jsonl(args.output, (proposal.to_dict() for proposal in ordered))
    _print_json({"proposal_count": len(ordered), "output": args.output})
    return 0


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
                raise ValueError(
                    f"Relation {relation.relation_id!r} has invalid evidence span"
                )
    ordered = sorted(relations, key=lambda item: item.relation_id)
    write_jsonl(args.output, (relation.to_dict() for relation in ordered))
    _print_json({"relation_count": len(ordered), "output": args.output})
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
    source_fingerprints = (
        ()
        if args.artifacts is None
        else tuple(
            artifact.object.sha256 for artifact in load_source_artifacts(args.artifacts)
        )
    )
    snapshot = SnapshotBuilder(
        split_config=SnapshotSplitConfig(
            development_fraction=args.development_fraction,
            challenge_sources=frozenset(args.challenge_source),
            challenge_templates=frozenset(args.challenge_template),
            hash_salt=args.hash_salt,
            max_synthetic_train_fraction=args.max_synthetic_fraction,
        ),
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
