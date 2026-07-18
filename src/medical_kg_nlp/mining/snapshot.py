"""Leakage-safe, deterministic, and atomic mined-dataset snapshot freezing."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from medical_kg_nlp.mining.catalog import ParquetSnapshotWriter
from medical_kg_nlp.mining.dedup import StableTextDeduplicator
from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.mining.policy import MiningQualityGate
from medical_kg_nlp.mining.quality import GoldAgreementGate, ReviewAgreementEvaluator
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    DatasetSnapshot,
    MinedDocument,
    RedistributionPolicy,
    RelationProposal,
    ReviewStatus,
)

__all__ = ["SnapshotBuilder", "SnapshotSplitConfig"]


@dataclass(frozen=True)
class SnapshotSplitConfig:
    """Frozen grouping and held-out rules for one dataset campaign."""

    development_fraction: float = 0.1
    challenge_sources: frozenset[str] = frozenset()
    challenge_templates: frozenset[str] = frozenset()
    hash_salt: str = "medical-kg-snapshot-v1"
    max_synthetic_train_fraction: float = 0.4

    def __post_init__(self) -> None:
        if not 0.0 <= self.development_fraction < 1.0:
            raise ValueError("development_fraction must be in [0, 1)")
        if not 0.0 <= self.max_synthetic_train_fraction <= 1.0:
            raise ValueError("max_synthetic_train_fraction must be in [0, 1]")
        if not self.hash_salt:
            raise ValueError("hash_salt must be non-empty")


class SnapshotBuilder:
    """Validate, split, and freeze one immutable dataset snapshot."""

    schema_version = "medical-dataset-snapshot.v1"

    def __init__(
        self,
        *,
        split_config: SnapshotSplitConfig | None = None,
        quality_gate: MiningQualityGate | None = None,
        deduplicator: StableTextDeduplicator | None = None,
        agreement_gate: GoldAgreementGate | None = None,
        rows_per_shard: int = 50_000,
    ) -> None:
        self.split_config = split_config or SnapshotSplitConfig()
        self.quality_gate = quality_gate or MiningQualityGate()
        self.deduplicator = deduplicator or StableTextDeduplicator()
        self.agreement_gate = agreement_gate
        self.rows_per_shard = rows_per_shard

    def freeze(
        self,
        *,
        version: str,
        created_at: str,
        output_dir: str | Path,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal] = (),
        relations: Sequence[RelationProposal] = (),
        source_fingerprints: Sequence[str] = (),
        write_parquet: bool = True,
    ) -> DatasetSnapshot:
        if not version.strip() or not created_at.strip():
            raise ValueError("Snapshot version and created_at must be explicit")
        ordered_documents = tuple(sorted(documents, key=lambda item: item.document_id))
        ordered_annotations = tuple(
            sorted(annotations, key=lambda item: item.annotation_id)
        )
        ordered_relations = tuple(sorted(relations, key=lambda item: item.relation_id))
        issues = list(self.quality_gate.validate(ordered_documents, ordered_annotations))
        issues.extend(
            _relation_issues(ordered_documents, ordered_annotations, ordered_relations)
        )
        splits, split_groups = self._assign_splits(ordered_documents)
        issues.extend(_challenge_issues(splits, ordered_annotations))
        issues.extend(_challenge_relation_issues(splits, ordered_relations))
        issues.extend(_synthetic_challenge_issues(splits, ordered_documents))
        issues.extend(
            _synthetic_fraction_issues(
                splits,
                ordered_documents,
                maximum=self.split_config.max_synthetic_train_fraction,
            )
        )
        agreement_report = None
        if self.agreement_gate is not None:
            agreement_report = ReviewAgreementEvaluator().evaluate(
                ordered_documents,
                ordered_annotations,
                ordered_relations,
            )
            issues.extend(
                self.agreement_gate.validate(
                    agreement_report,
                    has_gold_relations=any(
                        relation.layer
                        in {AnnotationLayer.GOLD, AnnotationLayer.CHALLENGE}
                        for relation in ordered_relations
                    ),
                )
            )
        if issues:
            raise ValueError("Snapshot validation failed:\n" + "\n".join(sorted(set(issues))))

        fingerprints = tuple(sorted(set(source_fingerprints)))
        restricted_reasons = _restricted_reasons(ordered_documents)
        split_counts = tuple(sorted(Counter(splits.values()).items()))
        core_manifest = {
            "schema_version": self.schema_version,
            "version": version,
            "created_at": created_at,
            "source_fingerprints": list(fingerprints),
            "counts": {
                "documents": len(ordered_documents),
                "annotations": len(ordered_annotations),
                "relations": len(ordered_relations),
            },
            "split_counts": dict(split_counts),
            "splits": dict(sorted(splits.items())),
            "split_groups": dict(sorted(split_groups.items())),
            "content_fingerprints": {
                "documents": _records_fingerprint(
                    document.to_dict() for document in ordered_documents
                ),
                "annotations": _records_fingerprint(
                    annotation.to_dict() for annotation in ordered_annotations
                ),
                "relations": _records_fingerprint(
                    relation.to_dict() for relation in ordered_relations
                ),
            },
            "redistributable": not restricted_reasons,
            "restricted_reasons": list(restricted_reasons),
            "quality": (
                None if agreement_report is None else agreement_report.to_dict()
            ),
            "storage": {
                "format": "parquet" if write_parquet else "manifest_only",
                "tables": _expected_tables(
                    len(ordered_documents),
                    len(ordered_annotations),
                    len(ordered_relations),
                    rows_per_shard=self.rows_per_shard,
                )
                if write_parquet
                else {},
            },
        }
        identity_hash = _mapping_fingerprint(core_manifest)
        snapshot_id = f"{_slug(version)}-{identity_hash[:16]}"
        manifest = {"snapshot_id": snapshot_id, **core_manifest}
        manifest_bytes = _pretty_json_bytes(manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        snapshot = DatasetSnapshot(
            snapshot_id=snapshot_id,
            version=version,
            manifest_sha256=manifest_sha256,
            document_count=len(ordered_documents),
            annotation_count=len(ordered_annotations),
            relation_count=len(ordered_relations),
            source_fingerprints=fingerprints,
            split_counts=split_counts,
            redistributable=not restricted_reasons,
            created_at=created_at,
            restricted_reasons=restricted_reasons,
        )
        self._write_snapshot(
            output_dir=Path(output_dir),
            manifest=manifest,
            snapshot=snapshot,
            documents=ordered_documents,
            annotations=ordered_annotations,
            relations=ordered_relations,
            splits=splits,
            write_parquet=write_parquet,
        )
        return snapshot

    def _assign_splits(
        self, documents: Sequence[MinedDocument]
    ) -> tuple[dict[str, str], dict[str, str]]:
        duplicate_groups = self.deduplicator.group(documents)
        components = _connected_groups(documents, duplicate_groups)
        splits: dict[str, str] = {}
        split_groups: dict[str, str] = {}
        by_id = {document.document_id: document for document in documents}
        for component_id, document_ids in sorted(components.items()):
            component_documents = [by_id[document_id] for document_id in document_ids]
            held_out = any(self._is_challenge(document) for document in component_documents)
            if held_out:
                split = "challenge"
            else:
                bucket = int(
                    hashlib.sha256(
                        f"{self.split_config.hash_salt}\0{component_id}".encode()
                    ).hexdigest()[:8],
                    16,
                ) / 0xFFFFFFFF
                split = (
                    "development"
                    if bucket < self.split_config.development_fraction
                    else "train"
                )
            for document_id in document_ids:
                splits[document_id] = split
                split_groups[document_id] = component_id
        return splits, split_groups

    def _is_challenge(self, document: MinedDocument) -> bool:
        source_id = document.metadata.get(
            "source_id", document.source_artifact_id.split(":", 1)[0]
        )
        template_id = document.metadata.get("template_id", "")
        return (
            source_id in self.split_config.challenge_sources
            or template_id in self.split_config.challenge_templates
        )

    def _write_snapshot(
        self,
        *,
        output_dir: Path,
        manifest: Mapping[str, object],
        snapshot: DatasetSnapshot,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
        relations: Sequence[RelationProposal],
        splits: Mapping[str, str],
        write_parquet: bool,
    ) -> None:
        if output_dir.exists():
            existing_manifest = output_dir / "manifest.json"
            if existing_manifest.is_file():
                existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
                if existing == manifest:
                    return
            raise FileExistsError(f"Immutable snapshot path already exists: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
        )
        try:
            if write_parquet:
                ParquetSnapshotWriter(
                    staging, rows_per_shard=self.rows_per_shard
                ).write(
                    documents=documents,
                    annotations=annotations,
                    relations=relations,
                    splits=splits,
                )
                # tables.json duplicates manifest information and would widen the contract.
                (staging / "tables.json").unlink(missing_ok=True)
            write_json(staging / "manifest.json", manifest)
            write_json(staging / "snapshot.json", snapshot.to_dict())
            staging.replace(output_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _connected_groups(
    documents: Sequence[MinedDocument], duplicate_groups: Mapping[str, str]
) -> dict[str, tuple[str, ...]]:
    ordered = sorted(documents, key=lambda item: item.document_id)
    parent = list(range(len(ordered)))
    first_by_group: dict[str, int] = {}
    for index, document in enumerate(ordered):
        group_keys = {
            duplicate_groups[document.document_id],
            *document.group_ids,
            *(
                f"{key}:{document.metadata[key]}"
                for key in ("patient_id", "case_id", "article_id", "template_id", "concept_family")
                if document.metadata.get(key)
            ),
        }
        for group_key in sorted(group_keys):
            previous = first_by_group.setdefault(group_key, index)
            _union(parent, index, previous)
    members: dict[int, list[str]] = defaultdict(list)
    for index, document in enumerate(ordered):
        members[_find(parent, index)].append(document.document_id)
    result: dict[str, tuple[str, ...]] = {}
    for document_ids in members.values():
        identity = "\n".join(sorted(document_ids))
        component_id = f"split-group:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        result[component_id] = tuple(sorted(document_ids))
    return result


def _challenge_issues(
    splits: Mapping[str, str], annotations: Sequence[AnnotationProposal]
) -> list[str]:
    issues: list[str] = []
    for annotation in annotations:
        if splits.get(annotation.document_id) != "challenge":
            continue
        if annotation.layer not in {AnnotationLayer.GOLD, AnnotationLayer.CHALLENGE}:
            issues.append(f"non_gold_challenge:{annotation.annotation_id}")
        if annotation.review_status is not ReviewStatus.ACCEPTED:
            issues.append(f"unreviewed_challenge:{annotation.annotation_id}")
    return issues


def _challenge_relation_issues(
    splits: Mapping[str, str], relations: Sequence[RelationProposal]
) -> list[str]:
    issues: list[str] = []
    for relation in relations:
        if splits.get(relation.document_id) != "challenge":
            continue
        if relation.layer not in {AnnotationLayer.GOLD, AnnotationLayer.CHALLENGE}:
            issues.append(f"non_gold_challenge_relation:{relation.relation_id}")
        if relation.review_status is not ReviewStatus.ACCEPTED:
            issues.append(f"unreviewed_challenge_relation:{relation.relation_id}")
        if relation.labeler_id is None:
            issues.append(f"anonymous_challenge_relation:{relation.relation_id}")
    return issues


def _synthetic_fraction_issues(
    splits: Mapping[str, str],
    documents: Sequence[MinedDocument],
    *,
    maximum: float,
) -> list[str]:
    training = [document for document in documents if splits[document.document_id] == "train"]
    if not training:
        return []
    synthetic_count = sum(
        document.metadata.get("origin") == "synthetic" for document in training
    )
    fraction = synthetic_count / len(training)
    if fraction > maximum:
        return [f"synthetic_train_fraction:{fraction:.6f}>{maximum:.6f}"]
    return []


def _synthetic_challenge_issues(
    splits: Mapping[str, str], documents: Sequence[MinedDocument]
) -> list[str]:
    return [
        f"synthetic_challenge_document:{document.document_id}"
        for document in documents
        if splits[document.document_id] == "challenge"
        and document.metadata.get("origin") == "synthetic"
    ]


def _relation_issues(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    relations: Sequence[RelationProposal],
) -> list[str]:
    documents_by_id = {document.document_id: document for document in documents}
    annotations_by_id = {annotation.annotation_id: annotation for annotation in annotations}
    duplicate_ids = [
        relation_id
        for relation_id, count in Counter(
            relation.relation_id for relation in relations
        ).items()
        if count > 1
    ]
    issues = [f"duplicate_relation:{relation_id}" for relation_id in duplicate_ids]
    for relation in relations:
        document = documents_by_id.get(relation.document_id)
        if document is None:
            issues.append(f"missing_relation_document:{relation.relation_id}")
            continue
        try:
            relation.validate(document, annotations_by_id)
        except ValueError as error:
            issues.append(f"relation:{relation.relation_id}:{error}")
    return issues


def _restricted_reasons(documents: Sequence[MinedDocument]) -> tuple[str, ...]:
    reasons: set[str] = set()
    restricted_access = {
        AccessClass.CREDENTIALLED,
        AccessClass.DUA,
        AccessClass.LOCAL_PRIVATE,
        AccessClass.QUARANTINE,
    }
    for document in documents:
        if document.access_class in restricted_access:
            reasons.add(f"access:{document.access_class.value}")
        if document.redistribution in {
            RedistributionPolicy.NON_COMMERCIAL,
            RedistributionPolicy.PROHIBITED,
            RedistributionPolicy.UNKNOWN,
        }:
            reasons.add(f"redistribution:{document.redistribution.value}")
    return tuple(sorted(reasons))


def _records_fingerprint(records: Iterable[Mapping[str, object]]) -> str:
    payload = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping_fingerprint(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pretty_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _expected_tables(
    document_count: int,
    annotation_count: int,
    relation_count: int,
    *,
    rows_per_shard: int,
) -> dict[str, list[str]]:
    return {
        name: [
            f"{name}/part-{index:05d}.parquet"
            for index in range((count + rows_per_shard - 1) // rows_per_shard)
        ]
        for name, count in (
            ("documents", document_count),
            ("annotations", annotation_count),
            ("relations", relation_count),
        )
    }


def _slug(value: str) -> str:
    result = "".join(character if character.isalnum() else "-" for character in value)
    return "-".join(part for part in result.split("-") if part).lower() or "snapshot"


def _find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: list[int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[max(left_root, right_root)] = min(left_root, right_root)
