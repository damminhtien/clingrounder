"""Thin CLI orchestration for task-neutral data mining operations."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import yaml

from medical_kg_nlp.mining.coverage import CoverageCubePlanner, CoverageTarget
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
from medical_kg_nlp.mining.policy import SourcePolicyGate
from medical_kg_nlp.mining.ports import ProposalLabelerPort
from medical_kg_nlp.mining.profile import (
    build_dataset_profile,
    profile_blocking_issue_count,
)
from medical_kg_nlp.mining.records import SourceRequest
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

__all__ = [
    "build_dataset",
    "export_review",
    "freeze_snapshot",
    "import_review",
    "inspect_dataset",
    "propose_labels",
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


def _load_mapping(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected mapping in {path}")
    return {str(key): value for key, value in raw.items()}


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
