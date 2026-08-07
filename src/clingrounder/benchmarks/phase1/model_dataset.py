"""Build leakage-safe five-type NER data from the frozen Phase 1 train split.

This module is benchmark-owned because it understands Phase 1 labels and the manually frozen
76/24 split. Generic mining and model code consume only the resulting neutral span dataset.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.benchmarks.phase1.manual_gold_mining import (
    Phase1ManualGoldMiningCorpus,
    load_phase1_manual_gold_mining_corpus,
)
from clingrounder.mining.dedup import StableTextDeduplicator
from clingrounder.mining.io import write_json, write_jsonl
from clingrounder.mining.model_dataset import SpanDatasetConfig, export_span_dataset
from clingrounder.mining.records import MinedDocument
from clingrounder.benchmarks.phase1.ontology import PHASE1_ENTITY_TYPE_RULES
from clingrounder.utils.hashing import sha256_file, sha256_text

__all__ = [
    "PHASE1_FIVE_TYPE_LABELS",
    "Phase1ModelDatasetConfig",
    "build_phase1_model_dataset",
    "build_phase1_model_splits",
]

_SCHEMA_VERSION = "phase1-five-type-model-dataset.v1"
PHASE1_FIVE_TYPE_LABELS = tuple(
    rule.internal_type.value for rule in PHASE1_ENTITY_TYPE_RULES
)


@dataclass(frozen=True)
class Phase1ModelDatasetConfig:
    """Pinned inputs and deterministic chunk/split controls for one model dataset."""

    input_dir: Path = Path("data/raw/input")
    gold_dir: Path = Path("data/manual_gold")
    frozen_split_manifest: Path = Path("data/manual_gold/holdout_manifest.json")
    public_spec_input: Path = Path("tests/fixtures/phase1/btc_medication_list_crlf.txt")
    public_spec_expected: Path = Path(
        "tests/fixtures/phase1/btc_medication_list_expected.json"
    )
    development_fraction: float = 0.2
    split_salt: str = "42"
    max_characters: int = 1600
    include_empty_chunks: bool = True
    empty_chunk_rate: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.development_fraction < 1.0:
            raise ValueError("development_fraction must be between zero and one")
        if not self.split_salt:
            raise ValueError("split_salt must be non-empty")
        SpanDatasetConfig(
            max_characters=self.max_characters,
            entity_types=PHASE1_FIVE_TYPE_LABELS,
            include_empty_chunks=self.include_empty_chunks,
            empty_chunk_rate=self.empty_chunk_rate,
        )


def build_phase1_model_dataset(
    output_dir: str | Path,
    *,
    config: Phase1ModelDatasetConfig | None = None,
) -> dict[str, Any]:
    """Build an atomic, portable five-type dataset without touching holdout or Round 2.

    The source loader verifies the complete frozen 76/24 corpus before selecting ``train``. This
    prevents a stale or edited gold directory from silently changing model supervision.
    """

    active = config or Phase1ModelDatasetConfig()
    corpus = load_phase1_manual_gold_mining_corpus(
        active.input_dir,
        active.gold_dir,
        active.frozen_split_manifest,
        split="train",
    )
    source_split = _load_source_split_contract(active.frozen_split_manifest)
    _validate_source_isolation(corpus, source_split)
    labels = {annotation.entity_type for annotation in corpus.annotations}
    if labels != set(PHASE1_FIVE_TYPE_LABELS):
        raise ValueError(
            "Phase 1 model data must contain exactly the five task labels: "
            f"observed={sorted(labels)}"
        )

    splits, split_groups, duplicate_counts = build_phase1_model_splits(
        corpus.documents,
        development_fraction=active.development_fraction,
        split_salt=active.split_salt,
    )
    split_counts = Counter(splits.values())
    if set(split_counts) != {"train", "development"}:
        raise ValueError("Phase 1 model data requires non-empty train and development splits")

    build_contract = _build_contract(active, corpus, source_split, splits)
    build_key = _mapping_sha256(build_contract)
    target = Path(output_dir)
    existing = _load_existing_build(target, build_key)
    if existing is not None:
        return existing

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        documents_path = staging / "documents.jsonl"
        annotations_path = staging / "annotations.jsonl"
        split_manifest_path = staging / "split_manifest.json"
        spans_path = staging / "spans.jsonl"
        dataset_manifest_path = staging / "manifest.json"

        write_jsonl(documents_path, (document.to_dict() for document in corpus.documents))
        write_jsonl(
            annotations_path,
            (annotation.to_dict() for annotation in corpus.annotations),
        )
        model_split_manifest = _model_split_manifest(
            corpus,
            source_split,
            splits,
            split_groups,
            duplicate_counts,
            active,
        )
        write_json(split_manifest_path, model_split_manifest)
        dataset_manifest = export_span_dataset(
            corpus.documents,
            corpus.annotations,
            splits,
            SpanDatasetConfig(
                max_characters=active.max_characters,
                entity_types=PHASE1_FIVE_TYPE_LABELS,
                include_empty_chunks=active.include_empty_chunks,
                empty_chunk_rate=active.empty_chunk_rate,
            ),
            output_path=spans_path,
            manifest_path=dataset_manifest_path,
            documents_path=documents_path,
            annotations_path=annotations_path,
            split_manifest_path=split_manifest_path,
            manifest_root=staging,
        )
        _validate_dataset_manifest(dataset_manifest, corpus, split_counts)
        output_hashes = {
            path.name: sha256_file(path)
            for path in (
                documents_path,
                annotations_path,
                split_manifest_path,
                spans_path,
                dataset_manifest_path,
            )
        }
        build_manifest = {
            "schema_version": _SCHEMA_VERSION,
            "build_key": build_key,
            "build_contract": build_contract,
            "dataset": {
                "document_count": len(corpus.documents),
                "annotation_count": len(corpus.annotations),
                "chunk_count": dataset_manifest["chunk_count"],
                "split_document_counts": dict(sorted(split_counts.items())),
                "split_chunk_counts": dataset_manifest["split_chunk_counts"],
                "entity_type_counts": dataset_manifest["entity_type_counts"],
            },
            "outputs": dict(sorted(output_hashes.items())),
        }
        write_json(staging / "build_manifest.json", build_manifest)
        staging.replace(target)
        return build_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_phase1_model_splits(
    documents: Sequence[MinedDocument],
    *,
    development_fraction: float,
    split_salt: str,
    deduplicator: StableTextDeduplicator | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    """Assign duplicate groups atomically to deterministic train/development splits."""

    if not documents:
        raise ValueError("Phase 1 model split requires documents")
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between zero and one")
    if not split_salt:
        raise ValueError("split_salt must be non-empty")
    groups = (deduplicator or StableTextDeduplicator()).describe_groups(documents)
    splits: dict[str, str] = {}
    split_groups: dict[str, str] = {}
    for group in groups:
        bucket = int(
            hashlib.sha256(f"{split_salt}\0{group.group_id}".encode()).hexdigest()[:8],
            16,
        ) / 0xFFFFFFFF
        split = "development" if bucket < development_fraction else "train"
        for document_id in group.document_ids:
            splits[document_id] = split
            split_groups[document_id] = group.group_id
    # INVARIANT: exact and near duplicate groups can never cross the model evaluation boundary.
    for group in groups:
        if len({splits[document_id] for document_id in group.document_ids}) != 1:
            raise RuntimeError(f"Duplicate group crossed model split: {group.group_id}")
    return (
        dict(sorted(splits.items())),
        dict(sorted(split_groups.items())),
        dict(sorted(Counter(group.kind.value for group in groups).items())),
    )


def _load_source_split_contract(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("assignments"), list):
        raise ValueError("Frozen Phase 1 split manifest requires assignments")
    by_split: dict[str, list[str]] = {"train": [], "holdout": []}
    for row in raw["assignments"]:
        if not isinstance(row, Mapping):
            raise ValueError("Frozen Phase 1 split assignment must be an object")
        source_id = str(row.get("document_id", ""))
        split = str(row.get("split", ""))
        if split not in by_split or not source_id:
            raise ValueError("Frozen Phase 1 split assignment is invalid")
        by_split[split].append(source_id)
    return {
        "manifest_sha256": sha256_file(path),
        "train_ids": tuple(sorted(by_split["train"], key=_source_sort_key)),
        "holdout_ids": tuple(sorted(by_split["holdout"], key=_source_sort_key)),
    }


def _validate_source_isolation(
    corpus: Phase1ManualGoldMiningCorpus,
    source_split: Mapping[str, Any],
) -> None:
    selected = {
        document.metadata.get("source_document_id", "") for document in corpus.documents
    }
    expected = set(source_split["train_ids"])
    holdout = set(source_split["holdout_ids"])
    if selected != expected:
        raise ValueError("Phase 1 model corpus does not equal the frozen train source IDs")
    if selected & holdout:
        raise ValueError("Phase 1 holdout document entered model supervision")
    if any(document.source_artifact_id.startswith("phase1_round2") for document in corpus.documents):
        raise ValueError("Round 2 documents cannot enter the supervised model dataset")


def _build_contract(
    config: Phase1ModelDatasetConfig,
    corpus: Phase1ManualGoldMiningCorpus,
    source_split: Mapping[str, Any],
    splits: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "source_corpus_fingerprint_sha256": corpus.corpus_fingerprint,
        "source_split_manifest_sha256": source_split["manifest_sha256"],
        "source_split": "train",
        "selected_source_document_ids_sha256": _ids_sha256(source_split["train_ids"]),
        "selected_source_document_count": len(source_split["train_ids"]),
        "excluded_holdout_document_ids_sha256": _ids_sha256(source_split["holdout_ids"]),
        "excluded_holdout_document_count": len(source_split["holdout_ids"]),
        "round2_included": False,
        "labels": sorted(PHASE1_FIVE_TYPE_LABELS),
        "development_fraction": config.development_fraction,
        "split_salt": config.split_salt,
        "max_characters": config.max_characters,
        "include_empty_chunks": config.include_empty_chunks,
        "empty_chunk_rate": config.empty_chunk_rate,
        "model_split_sha256": _mapping_sha256(splits),
        # The BTC sample validates conventions but is not an extra supervised document.
        "public_executable_spec": {
            "input_sha256": sha256_file(config.public_spec_input),
            "expected_sha256": sha256_file(config.public_spec_expected),
            "included_in_training": False,
            "runtime_lookup_memory": False,
        },
        "builder_sha256": sha256_file(Path(__file__)),
    }


def _model_split_manifest(
    corpus: Phase1ManualGoldMiningCorpus,
    source_split: Mapping[str, Any],
    splits: Mapping[str, str],
    split_groups: Mapping[str, str],
    duplicate_counts: Mapping[str, int],
    config: Phase1ModelDatasetConfig,
) -> dict[str, Any]:
    source_ids_by_split: dict[str, list[str]] = {"train": [], "development": []}
    for document in corpus.documents:
        source_id = document.metadata["source_document_id"]
        source_ids_by_split[splits[document.document_id]].append(source_id)
    return {
        "schema_version": "phase1-model-training-split.v1",
        "source_corpus_fingerprint_sha256": corpus.corpus_fingerprint,
        "source_split_manifest_sha256": source_split["manifest_sha256"],
        "source_split": "train",
        "split_policy": {
            "algorithm": "sha256_duplicate_group_bucket",
            "development_fraction": config.development_fraction,
            "hash_salt": config.split_salt,
            "deduplicator": "StableTextDeduplicator",
        },
        "splits": dict(sorted(splits.items())),
        "split_groups": dict(sorted(split_groups.items())),
        "source_document_ids": {
            name: sorted(values, key=_source_sort_key)
            for name, values in source_ids_by_split.items()
        },
        "duplicate_group_counts": dict(sorted(duplicate_counts.items())),
        "excluded_holdout": {
            "document_count": len(source_split["holdout_ids"]),
            "document_ids_sha256": _ids_sha256(source_split["holdout_ids"]),
        },
        "round2_included": False,
    }


def _validate_dataset_manifest(
    manifest: Mapping[str, Any],
    corpus: Phase1ManualGoldMiningCorpus,
    split_counts: Mapping[str, int],
) -> None:
    if int(manifest["document_count"]) != len(corpus.documents):
        raise RuntimeError("Span dataset dropped source documents")
    if int(manifest["entity_count"]) != len(corpus.annotations):
        raise RuntimeError("Span dataset dropped source annotations")
    if set(manifest["entity_type_counts"]) != set(PHASE1_FIVE_TYPE_LABELS):
        raise RuntimeError("Span dataset lost one or more Phase 1 labels")
    if set(split_counts) != {"train", "development"}:
        raise RuntimeError("Span dataset is missing a model split")


def _load_existing_build(target: Path, build_key: str) -> dict[str, Any] | None:
    if not target.exists():
        return None
    manifest_path = target / "build_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"Dataset output exists without a build manifest: {target}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("build_key") != build_key:
        raise FileExistsError(f"Dataset output belongs to a different build: {target}")
    outputs = raw.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Existing Phase 1 model build has no output fingerprints")
    for relative_path, expected_sha256 in outputs.items():
        path = target / str(relative_path)
        if not path.is_file() or sha256_file(path) != str(expected_sha256):
            raise ValueError(f"Existing Phase 1 model output failed fingerprint check: {path}")
    return {str(key): value for key, value in raw.items()}


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def _ids_sha256(values: Sequence[str]) -> str:
    return sha256_text("\n".join(values))


def _source_sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)
