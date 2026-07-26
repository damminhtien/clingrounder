"""Build leakage-safe Q&A and educational NER training views.

The augmentation changes discourse framing only. It copies reviewed train spans into a child
document with new immutable coordinates; it never invents medical labels, opens development
labels for augmentation, or consumes Round 2/quarantined artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.training.span_dataset import (
    SpanTrainingRecord,
    scan_span_dataset,
    validate_span_dataset_manifest,
)
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text

__all__ = [
    "Phase1RegionAugmentationConfig",
    "RegionAugmentationKind",
    "build_phase1_region_augmented_dataset",
]

RegionAugmentationKind = Literal["question_answer", "educational"]

_SCHEMA_VERSION = "phase1-region-augmented-model-dataset.v1"
_RENDERER_VERSION = "phase1-region-renderer.v1"
_RENDER_TEMPLATES: dict[RegionAugmentationKind, tuple[str, str]] = {
    "question_answer": (
        "Câu hỏi của người dùng: Những thông tin y khoa nào được nhắc đến "
        "trong nội dung sau?\n",
        "\nCâu trả lời của bác sĩ: Các thực thể cần nhận diện nằm trong nội dung trên.",
    ),
    "educational": (
        "Thông tin giáo dục sức khỏe cần lưu ý:\n",
        "\nNội dung trên được trình bày nhằm nhận diện các thuật ngữ y khoa.",
    ),
}
_DISALLOWED_SOURCE_MARKERS = ("round2", "leak", "quarantine")


@dataclass(frozen=True)
class Phase1RegionAugmentationConfig:
    """Pinned source dataset and bounded train-only augmentation controls."""

    source_dataset_path: Path
    source_manifest_path: Path
    source_build_manifest_path: Path
    max_synthetic_train_fraction: float = 0.4
    seed: str = "phase1-qa-educational-v1"
    train_split: str = "train"
    development_split: str = "development"
    render_kinds: tuple[RegionAugmentationKind, ...] = (
        "question_answer",
        "educational",
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_synthetic_train_fraction <= 0.4:
            raise ValueError("Synthetic train fraction must be in [0, 0.4]")
        if not self.seed.strip():
            raise ValueError("Augmentation seed must be non-empty")
        if not self.train_split or not self.development_split:
            raise ValueError("Train and development split names must be non-empty")
        if self.train_split == self.development_split:
            raise ValueError("Train and development split names must differ")
        if not self.render_kinds:
            raise ValueError("At least one region augmentation kind is required")
        if len(self.render_kinds) != len(set(self.render_kinds)):
            raise ValueError("Region augmentation kinds must be unique")
        unknown = set(self.render_kinds) - set(_RENDER_TEMPLATES)
        if unknown:
            raise ValueError(f"Unsupported region augmentation kinds: {sorted(unknown)}")


def build_phase1_region_augmented_dataset(
    output_dir: str | Path,
    *,
    config: Phase1RegionAugmentationConfig,
) -> dict[str, Any]:
    """Create an atomic span dataset with bounded train-only style augmentation."""

    source_rows, source_records = _load_and_validate_source(config)
    build_contract = _build_contract(config, source_records)
    build_key = _mapping_sha256(build_contract)
    target = Path(output_dir)
    existing = _load_existing_build(target, build_key)
    if existing is not None:
        return existing

    train_records = [
        record for record in source_records if record.split == config.train_split
    ]
    development_records = [
        record for record in source_records if record.split == config.development_split
    ]
    if not train_records or not development_records:
        raise ValueError("Source dataset requires non-empty train and development splits")

    selected = _select_train_records(train_records, config)
    synthetic_rows = [
        _render_child_record(
            record,
            kind=config.render_kinds[index % len(config.render_kinds)],
            seed=config.seed,
        )
        for index, record in enumerate(selected)
    ]
    output_rows = [*source_rows, *synthetic_rows]

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        spans_path = staging / "spans.jsonl"
        output_sha256 = write_jsonl(spans_path, output_rows)
        summary = scan_span_dataset(spans_path)
        _validate_output_summary(
            summary.split_record_counts,
            source_records=source_records,
            synthetic_count=len(synthetic_rows),
            config=config,
        )
        type_counts: Counter[str] = Counter(
            str(entity["label"])
            for row in output_rows
            for entity in _entity_mappings(row)
        )
        manifest = {
            "schema_version": "mined-span-dataset.v1",
            "document_count": len(
                {str(row["document_id"]) for row in output_rows}
            ),
            "chunk_count": summary.record_count,
            "empty_chunk_count": summary.empty_record_count,
            "entity_count": summary.entity_count,
            "split_chunk_counts": dict(summary.split_record_counts),
            "entity_type_counts": dict(sorted(type_counts.items())),
            "config": {
                "augmentation_renderer_version": _RENDERER_VERSION,
                "max_synthetic_train_fraction": (
                    config.max_synthetic_train_fraction
                ),
                "render_kinds": list(config.render_kinds),
                "seed": config.seed,
            },
            "inputs": {
                "source_dataset": _fingerprinted_path(config.source_dataset_path),
                "source_manifest": _fingerprinted_path(config.source_manifest_path),
                "source_build_manifest": _fingerprinted_path(
                    config.source_build_manifest_path
                ),
            },
            "output": "spans.jsonl",
            "output_sha256": output_sha256,
            "augmentation": {
                "source_record_count": len(source_records),
                "source_train_record_count": len(train_records),
                "source_development_record_count": len(development_records),
                "synthetic_train_record_count": len(synthetic_rows),
                "synthetic_train_fraction": _synthetic_fraction(
                    len(train_records), len(synthetic_rows)
                ),
                "development_augmented": False,
                "round2_included": False,
                "quarantined_data_included": False,
            },
        }
        manifest_path = staging / "manifest.json"
        write_json(manifest_path, manifest)
        validate_span_dataset_manifest(spans_path, manifest_path, summary)

        output_hashes = {
            "manifest.json": sha256_file(manifest_path),
            "spans.jsonl": sha256_file(spans_path),
        }
        build_manifest = {
            "schema_version": _SCHEMA_VERSION,
            "build_key": build_key,
            "build_contract": build_contract,
            "dataset": {
                "record_count": summary.record_count,
                "entity_count": summary.entity_count,
                "split_record_counts": dict(summary.split_record_counts),
                "split_entity_counts": dict(summary.split_entity_counts),
                "synthetic_train_record_count": len(synthetic_rows),
                "synthetic_train_fraction": _synthetic_fraction(
                    len(train_records), len(synthetic_rows)
                ),
            },
            "outputs": output_hashes,
        }
        write_json(staging / "build_manifest.json", build_manifest)
        staging.replace(target)
        return build_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_and_validate_source(
    config: Phase1RegionAugmentationConfig,
) -> tuple[list[dict[str, Any]], tuple[SpanTrainingRecord, ...]]:
    summary = scan_span_dataset(config.source_dataset_path)
    validate_span_dataset_manifest(
        config.source_dataset_path,
        config.source_manifest_path,
        summary,
    )
    build_manifest = json.loads(
        config.source_build_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(build_manifest, Mapping):
        raise ValueError("Source build manifest must be an object")
    build_contract = build_manifest.get("build_contract")
    if not isinstance(build_contract, Mapping):
        raise ValueError("Source build manifest has no build contract")
    if build_contract.get("round2_included") is not False:
        raise ValueError("Source model dataset must explicitly exclude Round 2")
    outputs = build_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Source build manifest has no output fingerprints")
    if outputs.get("spans.jsonl") != summary.dataset_sha256:
        raise ValueError("Source dataset differs from its build manifest")

    rows: list[dict[str, Any]] = []
    records: list[SpanTrainingRecord] = []
    with config.source_dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"{config.source_dataset_path}:{line_number}: row must be an object"
                )
            row = {str(key): value for key, value in raw.items()}
            record = SpanTrainingRecord.from_mapping(row)
            source_identity = (
                f"{record.source_artifact_id}\0{record.document_id}".lower()
            )
            if any(marker in source_identity for marker in _DISALLOWED_SOURCE_MARKERS):
                raise ValueError(
                    f"Disallowed supervision source for record {record.record_id!r}"
                )
            rows.append(row)
            records.append(record)
    return rows, tuple(records)


def _select_train_records(
    train_records: Sequence[SpanTrainingRecord],
    config: Phase1RegionAugmentationConfig,
) -> tuple[SpanTrainingRecord, ...]:
    eligible = [
        record
        for record in train_records
        if record.language.casefold().startswith("vi") and record.entities
    ]
    eligible.sort(
        key=lambda record: (
            hashlib.sha256(
                f"{config.seed}\0{record.record_id}".encode("utf-8")
            ).hexdigest(),
            record.record_id,
        )
    )
    maximum = _maximum_synthetic_records(
        len(train_records),
        config.max_synthetic_train_fraction,
    )
    return tuple(eligible[:maximum])


def _render_child_record(
    parent: SpanTrainingRecord,
    *,
    kind: RegionAugmentationKind,
    seed: str,
) -> dict[str, Any]:
    prefix, suffix = _RENDER_TEMPLATES[kind]
    text = f"{prefix}{parent.text}{suffix}"
    shift = len(prefix)
    identity = f"{_RENDERER_VERSION}\0{seed}\0{kind}\0{parent.record_id}"
    child_digest = sha256_text(identity)
    document_id = f"phase1-region-augmentation:{child_digest[:24]}"
    entities = [
        {
            "annotation_id": (
                "phase1-region-annotation:"
                + sha256_text(f"{child_digest}\0{entity.annotation_id}")[:24]
            ),
            "start": entity.start + shift,
            "end": entity.end + shift,
            "source_start": entity.start + shift,
            "source_end": entity.end + shift,
            "text": entity.text,
            "label": entity.label,
        }
        for entity in parent.entities
    ]
    row = {
        "record_id": f"phase1-region-record:{child_digest[:24]}",
        "document_id": document_id,
        "split": parent.split,
        "text": text,
        "text_sha256": sha256_text(text),
        "source_span": [0, len(text)],
        "language": parent.language,
        "note_type": kind,
        "source_artifact_id": f"synthetic:{_RENDERER_VERSION}",
        "entities": entities,
        "metadata": {
            "origin": "synthetic",
            "parent_document_id": parent.document_id,
            "parent_record_id": parent.record_id,
            "render_kind": kind,
            "renderer_version": _RENDERER_VERSION,
        },
    }
    # INVARIANT: synthetic children own a new immutable text coordinate space. Parent offsets are
    # provenance only and are never reused as child source offsets.
    SpanTrainingRecord.from_mapping(row)
    return row


def _maximum_synthetic_records(
    original_train_count: int,
    maximum_fraction: float,
) -> int:
    if original_train_count < 1 or maximum_fraction <= 0.0:
        return 0
    return math.floor(
        maximum_fraction * original_train_count / (1.0 - maximum_fraction)
    )


def _synthetic_fraction(original_count: int, synthetic_count: int) -> float:
    total = original_count + synthetic_count
    return 0.0 if total == 0 else synthetic_count / total


def _validate_output_summary(
    split_record_counts: Mapping[str, int],
    *,
    source_records: Sequence[SpanTrainingRecord],
    synthetic_count: int,
    config: Phase1RegionAugmentationConfig,
) -> None:
    source_counts = Counter(record.split for record in source_records)
    if split_record_counts.get(config.development_split, 0) != source_counts[
        config.development_split
    ]:
        raise RuntimeError("Development records changed during augmentation")
    expected_train = source_counts[config.train_split] + synthetic_count
    if split_record_counts.get(config.train_split, 0) != expected_train:
        raise RuntimeError("Unexpected augmented train record count")
    fraction = _synthetic_fraction(source_counts[config.train_split], synthetic_count)
    if fraction > config.max_synthetic_train_fraction:
        raise RuntimeError("Synthetic train fraction exceeds the configured cap")


def _build_contract(
    config: Phase1RegionAugmentationConfig,
    source_records: Sequence[SpanTrainingRecord],
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "renderer_version": _RENDERER_VERSION,
        "builder_sha256": sha256_file(Path(__file__)),
        "source_dataset_sha256": sha256_file(config.source_dataset_path),
        "source_manifest_sha256": sha256_file(config.source_manifest_path),
        "source_build_manifest_sha256": sha256_file(
            config.source_build_manifest_path
        ),
        "source_record_count": len(source_records),
        "seed": config.seed,
        "train_split": config.train_split,
        "development_split": config.development_split,
        "max_synthetic_train_fraction": config.max_synthetic_train_fraction,
        "render_kinds": list(config.render_kinds),
        "development_augmented": False,
        "round2_included": False,
        "quarantined_data_included": False,
        "label_generation": "copy_reviewed_parent_spans_only",
    }


def _load_existing_build(
    target: Path,
    build_key: str,
) -> dict[str, Any] | None:
    if not target.exists():
        return None
    manifest_path = target / "build_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(
            f"Augmented dataset exists without a build manifest: {target}"
        )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("build_key") != build_key:
        raise FileExistsError(
            f"Augmented dataset belongs to a different build: {target}"
        )
    outputs = raw.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Existing augmented dataset has no output fingerprints")
    for relative_path, expected_sha256 in outputs.items():
        path = target / str(relative_path)
        if not path.is_file() or sha256_file(path) != str(expected_sha256):
            raise ValueError(f"Augmented dataset fingerprint check failed: {path}")
    return {str(key): value for key, value in raw.items()}


def _entity_mappings(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entities = row.get("entities")
    if not isinstance(entities, list) or not all(
        isinstance(entity, Mapping) for entity in entities
    ):
        raise ValueError("Span record entities must be object lists")
    return entities


def _fingerprinted_path(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)
