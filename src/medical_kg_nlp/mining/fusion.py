"""Deterministic fusion of mined corpora into one leakage-safe data plane.

Exact raw-text duplicates are reconciled because their annotations share a character
coordinate system. Normalized and SimHash-near duplicates are only assigned a shared
split group; their documents and annotations remain independent.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from medical_kg_nlp.mining.dedup import (
    DuplicateGroup,
    StableTextDeduplicator,
)
from medical_kg_nlp.mining.io import (
    load_annotations,
    load_documents,
    load_relations,
    write_json,
    write_jsonl,
)
from medical_kg_nlp.mining.policy import MiningQualityGate
from medical_kg_nlp.mining.reconciliation import (
    DocumentCanonicalMapping,
    DuplicateReconciliationReport,
    reconcile_exact_duplicates,
)
from medical_kg_nlp.mining.records import (
    AnnotationProposal,
    MinedDocument,
    RelationProposal,
)
from medical_kg_nlp.utils.hashing import sha256_file

__all__ = [
    "CorpusFusionPlan",
    "CorpusFusionResult",
    "CorpusPartition",
    "FusionRunResult",
    "FusionSourcePlan",
    "RejectedRelation",
    "fuse_corpora",
    "load_corpus_fusion_plan",
    "run_corpus_fusion_plan",
]


class FusionSourcePlan(BaseModel):
    """Files belonging to one source-specific corpus partition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    documents: str = Field(min_length=1)
    annotations: str | None = None
    relations: str | None = None


class CorpusFusionPlan(BaseModel):
    """Strict declarative input for a reproducible corpus-fusion run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["medical-corpus-fusion-plan.v1"]
    output_root: str = Field(min_length=1)
    run_label: str = Field(default="open-corpus", pattern=r"^[a-z0-9][a-z0-9-]*$")
    hamming_threshold: int = Field(default=3, ge=0, le=16)
    bands: int = Field(default=4, gt=0)
    sources: tuple[FusionSourcePlan, ...]


@dataclass(frozen=True)
class CorpusPartition:
    """Validated in-memory records imported from one provenance boundary."""

    source_id: str
    documents: tuple[MinedDocument, ...]
    annotations: tuple[AnnotationProposal, ...] = ()
    relations: tuple[RelationProposal, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("Corpus source_id must be non-empty")


@dataclass(frozen=True)
class RejectedRelation:
    """A relation retained for audit after its endpoint was not materialized."""

    reason: str
    relation: RelationProposal

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "relation": self.relation.to_dict()}


@dataclass(frozen=True)
class CorpusFusionResult:
    """Fused records plus every artifact needed to audit data loss."""

    documents: tuple[MinedDocument, ...]
    annotations: tuple[AnnotationProposal, ...]
    review_annotations: tuple[AnnotationProposal, ...]
    relations: tuple[RelationProposal, ...]
    rejected_relations: tuple[RejectedRelation, ...]
    document_mappings: tuple[DocumentCanonicalMapping, ...]
    duplicate_groups: tuple[DuplicateGroup, ...]
    exact_reconciliation: DuplicateReconciliationReport
    report: dict[str, Any]


class FusionRunResult(BaseModel):
    """Stable paths and counts returned by an idempotent fusion run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    output_dir: str
    manifest: str
    document_count: int
    annotation_count: int
    review_annotation_count: int
    relation_count: int
    rejected_relation_count: int
    cache_hit: bool


def load_corpus_fusion_plan(path: str | Path) -> CorpusFusionPlan:
    """Load a strict fusion plan and resolve paths relative to that plan."""

    plan_path = Path(path).resolve()
    raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan = CorpusFusionPlan.model_validate(raw)
    if not plan.sources:
        raise ValueError("Corpus fusion requires at least one source")
    source_ids = [source.source_id for source in plan.sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Corpus fusion source_id values must be unique")
    if 64 % plan.bands:
        raise ValueError("Corpus fusion bands must be a divisor of 64")
    base = plan_path.parent
    sources = tuple(
        source.model_copy(
            update={
                "documents": str(_resolve_path(source.documents, base)),
                "annotations": (
                    None
                    if source.annotations is None
                    else str(_resolve_path(source.annotations, base))
                ),
                "relations": (
                    None
                    if source.relations is None
                    else str(_resolve_path(source.relations, base))
                ),
            }
        )
        for source in plan.sources
    )
    return plan.model_copy(
        update={
            "output_root": str(_resolve_path(plan.output_root, base)),
            "sources": sources,
        }
    )


def fuse_corpora(
    corpora: Sequence[CorpusPartition],
    *,
    deduplicator: StableTextDeduplicator | None = None,
) -> CorpusFusionResult:
    """Validate, reconcile, and group source partitions without offset remapping."""

    if not corpora:
        raise ValueError("At least one corpus partition is required")
    quality_gate = MiningQualityGate()
    all_documents: list[MinedDocument] = []
    all_annotations: list[AnnotationProposal] = []
    all_relations: list[RelationProposal] = []
    source_reports: list[dict[str, Any]] = []
    for corpus in corpora:
        issues = quality_gate.validate(corpus.documents, corpus.annotations)
        issues += tuple(_relation_issues(corpus.documents, corpus.annotations, corpus.relations))
        if issues:
            raise ValueError(
                f"Invalid corpus partition {corpus.source_id!r}:\n" + "\n".join(issues)
            )
        source_reports.append(
            {
                "source_id": corpus.source_id,
                "document_count": len(corpus.documents),
                "annotation_count": len(corpus.annotations),
                "relation_count": len(corpus.relations),
                "annotated_document_count": len(
                    {annotation.document_id for annotation in corpus.annotations}
                ),
            }
        )
        all_documents.extend(corpus.documents)
        all_annotations.extend(corpus.annotations)
        all_relations.extend(corpus.relations)

    global_issues = quality_gate.validate(all_documents, all_annotations)
    global_issues += tuple(_duplicate_relation_issues(all_relations))
    if global_issues:
        raise ValueError("Corpus partitions collide:\n" + "\n".join(global_issues))

    reconciled = reconcile_exact_duplicates(all_documents, all_annotations)
    deduplicator = deduplicator or StableTextDeduplicator()
    duplicate_groups = deduplicator.describe_groups(reconciled.documents)
    split_groups = {
        document_id: group.group_id
        for group in duplicate_groups
        if len(group.document_ids) > 1
        for document_id in group.document_ids
    }
    documents = tuple(
        replace(
            document,
            group_ids=tuple(
                sorted(
                    {
                        *document.group_ids,
                        *(
                            (split_groups[document.document_id],)
                            if document.document_id in split_groups
                            else ()
                        ),
                    }
                )
            ),
        )
        for document in reconciled.documents
    )
    relations, rejected_relations = _retain_relations(
        all_relations,
        reconciled.document_mappings,
        reconciled.training_annotations,
    )
    output_issues = quality_gate.validate(
        documents,
        (*reconciled.training_annotations, *reconciled.review_annotations),
    )
    output_issues += tuple(
        _relation_issues(documents, reconciled.training_annotations, relations)
    )
    if output_issues:
        raise ValueError("Corpus fusion produced invalid records:\n" + "\n".join(output_issues))

    group_counts = Counter(
        group.kind.value for group in duplicate_groups if len(group.document_ids) > 1
    )
    grouped_document_counts: Counter[str] = Counter()
    for group in duplicate_groups:
        if len(group.document_ids) > 1:
            grouped_document_counts[group.kind.value] += len(group.document_ids)
    report = {
        "schema_version": "medical-corpus-fusion-report.v1",
        "sources": sorted(source_reports, key=lambda item: item["source_id"]),
        "input": {
            "document_count": len(all_documents),
            "annotation_count": len(all_annotations),
            "relation_count": len(all_relations),
        },
        "output": {
            "document_count": len(documents),
            "annotation_count": len(reconciled.training_annotations),
            "review_annotation_count": len(reconciled.review_annotations),
            "relation_count": len(relations),
            "rejected_relation_count": len(rejected_relations),
        },
        "deduplication": {
            "raw_exact_group_count": reconciled.report.duplicate_group_count,
            "raw_exact_document_count": reconciled.report.duplicate_document_count,
            "split_only_group_counts": dict(sorted(group_counts.items())),
            "split_only_document_counts": dict(sorted(grouped_document_counts.items())),
        },
        "exact_reconciliation": reconciled.report.to_dict(),
        "validation": {
            "input_issue_count": 0,
            "output_issue_count": 0,
        },
    }
    return CorpusFusionResult(
        documents=documents,
        annotations=reconciled.training_annotations,
        review_annotations=reconciled.review_annotations,
        relations=relations,
        rejected_relations=rejected_relations,
        document_mappings=reconciled.document_mappings,
        duplicate_groups=duplicate_groups,
        exact_reconciliation=reconciled.report,
        report=report,
    )


def run_corpus_fusion_plan(path: str | Path) -> FusionRunResult:
    """Execute one plan into a content-addressed, immutable output directory."""

    plan = load_corpus_fusion_plan(path)
    inputs = tuple(_fingerprinted_source(source) for source in plan.sources)
    identity = {
        "schema_version": plan.schema_version,
        "hamming_threshold": plan.hamming_threshold,
        "bands": plan.bands,
        # SCALING: cache identity follows content, not checkout or mount paths.
        "sources": tuple(_source_content_identity(item) for item in inputs),
    }
    fingerprint = _json_sha256(identity)
    run_id = f"{plan.run_label}-{fingerprint[:12]}"
    output_dir = Path(plan.output_root) / run_id
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_fingerprint") != fingerprint:
            raise ValueError(f"Stale fusion output at {output_dir}")
        _validate_cached_outputs(manifest, output_dir)
        return _run_result_from_manifest(manifest, output_dir, cache_hit=True)
    if output_dir.exists():
        raise FileExistsError(f"Incomplete immutable fusion output exists: {output_dir}")

    corpora = tuple(
        CorpusPartition(
            source_id=source.source_id,
            documents=load_documents(source.documents),
            annotations=(
                () if source.annotations is None else load_annotations(source.annotations)
            ),
            relations=(() if source.relations is None else load_relations(source.relations)),
        )
        for source in plan.sources
    )
    result = fuse_corpora(
        corpora,
        deduplicator=StableTextDeduplicator(
            hamming_threshold=plan.hamming_threshold,
            bands=plan.bands,
        ),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_dir.parent))
    try:
        hashes = {
            "documents": write_jsonl(
                staging / "documents.jsonl",
                (document.to_dict() for document in result.documents),
            ),
            "annotations": write_jsonl(
                staging / "annotations.jsonl",
                (annotation.to_dict() for annotation in result.annotations),
            ),
            "review_annotations": write_jsonl(
                staging / "review_annotations.jsonl",
                (annotation.to_dict() for annotation in result.review_annotations),
            ),
            "relations": write_jsonl(
                staging / "relations.jsonl",
                (relation.to_dict() for relation in result.relations),
            ),
            "rejected_relations": write_jsonl(
                staging / "rejected_relations.jsonl",
                (relation.to_dict() for relation in result.rejected_relations),
            ),
            "document_map": write_jsonl(
                staging / "document_map.jsonl",
                (mapping.to_dict() for mapping in result.document_mappings),
            ),
            "duplicate_groups": write_jsonl(
                staging / "duplicate_groups.jsonl",
                (group.to_dict() for group in result.duplicate_groups),
            ),
            "fusion_report": write_json(staging / "fusion_report.json", result.report),
        }
        manifest = {
            "schema_version": "medical-corpus-fusion-manifest.v1",
            "run_id": run_id,
            "run_fingerprint": fingerprint,
            "config": {
                "hamming_threshold": plan.hamming_threshold,
                "bands": plan.bands,
            },
            "inputs": list(inputs),
            "outputs": {
                name: {"path": _output_filename(name), "sha256": digest}
                for name, digest in sorted(hashes.items())
            },
            "counts": result.report["output"],
        }
        write_json(staging / "manifest.json", manifest)
        # SCALING: all large JSONL files are complete before the directory becomes visible.
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _run_result_from_manifest(manifest, output_dir, cache_hit=False)


def _retain_relations(
    relations: Sequence[RelationProposal],
    mappings: Sequence[DocumentCanonicalMapping],
    annotations: Sequence[AnnotationProposal],
) -> tuple[tuple[RelationProposal, ...], tuple[RejectedRelation, ...]]:
    canonical_by_document = {
        mapping.document_id: mapping.canonical_document_id for mapping in mappings
    }
    annotation_ids = {annotation.annotation_id for annotation in annotations}
    accepted: list[RelationProposal] = []
    rejected: list[RejectedRelation] = []
    for relation in sorted(relations, key=lambda item: item.relation_id):
        canonical_id = canonical_by_document[relation.document_id]
        if canonical_id != relation.document_id:
            rejected.append(RejectedRelation("document_collapsed", relation))
        elif (
            relation.head_annotation_id not in annotation_ids
            or relation.tail_annotation_id not in annotation_ids
        ):
            # INVARIANT: never guess a relation endpoint after consensus creates new IDs.
            rejected.append(RejectedRelation("endpoint_not_materialized", relation))
        else:
            accepted.append(relation)
    return tuple(accepted), tuple(rejected)


def _relation_issues(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    relations: Sequence[RelationProposal],
) -> list[str]:
    documents_by_id = {document.document_id: document for document in documents}
    annotations_by_id = {annotation.annotation_id: annotation for annotation in annotations}
    issues = _duplicate_relation_issues(relations)
    for relation in relations:
        document = documents_by_id.get(relation.document_id)
        if document is None:
            issues.append(f"missing_relation_document:{relation.relation_id}")
            continue
        try:
            relation.validate(document, annotations_by_id)
        except ValueError as error:
            issues.append(f"relation:{relation.relation_id}:{error}")
    return sorted(issues)


def _duplicate_relation_issues(relations: Sequence[RelationProposal]) -> list[str]:
    counts = Counter(relation.relation_id for relation in relations)
    return sorted(
        f"duplicate_relation:{relation_id}"
        for relation_id, count in counts.items()
        if count > 1
    )


def _fingerprinted_source(source: FusionSourcePlan) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "documents": _fingerprinted_path(source.documents),
        "annotations": (
            None if source.annotations is None else _fingerprinted_path(source.annotations)
        ),
        "relations": (
            None if source.relations is None else _fingerprinted_path(source.relations)
        ),
    }


def _fingerprinted_path(path: str | Path) -> dict[str, str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": str(source), "sha256": sha256_file(source)}


def _source_content_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": source["source_id"],
        "documents_sha256": source["documents"]["sha256"],
        "annotations_sha256": (
            None if source["annotations"] is None else source["annotations"]["sha256"]
        ),
        "relations_sha256": (
            None if source["relations"] is None else source["relations"]["sha256"]
        ),
    }


def _validate_cached_outputs(manifest: Mapping[str, Any], output_dir: Path) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Fusion manifest has no outputs mapping")
    for name, raw in outputs.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Fusion output metadata {name!r} must be an object")
        path = output_dir / str(raw["path"])
        expected = str(raw["sha256"])
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Corrupt cached fusion output: {path}")


def _run_result_from_manifest(
    manifest: Mapping[str, Any], output_dir: Path, *, cache_hit: bool
) -> FusionRunResult:
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("Fusion manifest has no counts mapping")
    return FusionRunResult(
        run_id=str(manifest["run_id"]),
        output_dir=str(output_dir),
        manifest=str(output_dir / "manifest.json"),
        document_count=int(counts["document_count"]),
        annotation_count=int(counts["annotation_count"]),
        review_annotation_count=int(counts["review_annotation_count"]),
        relation_count=int(counts["relation_count"]),
        rejected_relation_count=int(counts["rejected_relation_count"]),
        cache_hit=cache_hit,
    )


def _output_filename(name: str) -> str:
    return {
        "documents": "documents.jsonl",
        "annotations": "annotations.jsonl",
        "review_annotations": "review_annotations.jsonl",
        "relations": "relations.jsonl",
        "rejected_relations": "rejected_relations.jsonl",
        "document_map": "document_map.jsonl",
        "duplicate_groups": "duplicate_groups.jsonl",
        "fusion_report": "fusion_report.json",
    }[name]


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
