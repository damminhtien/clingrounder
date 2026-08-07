"""Combine final authorized supervision with bounded Q&A/educational augmentation.

The 200-document final corpus remains the authoritative clinical supervision. This builder adds
only provenance-checked synthetic Q&A/educational records so the token model can learn genres
that the competition notes underrepresent, without allowing Round 2 or Friend31 data into fit.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.mining.io import write_json, write_jsonl
from clingrounder.training.span_dataset import (
    scan_span_dataset,
    validate_span_dataset_manifest,
)
from clingrounder.utils.hashing import sha256_file, sha256_text
from clingrounder.utils.io import read_jsonl

__all__ = [
    "Phase1FinalTokenTrainingBundleConfig",
    "build_phase1_final_token_training_bundle",
]

_SCHEMA_VERSION = "phase1-final-token-training-bundle.v1"
_SYNTHETIC_SOURCE_PREFIX = "synthetic:phase1-region-renderer.v1"
_ALLOWED_SYNTHETIC_NOTE_TYPES = frozenset({"educational", "question_answer"})


@dataclass(frozen=True, slots=True)
class Phase1FinalTokenTrainingBundleConfig:
    """Pinned sources and a hard cap for synthetic token-classifier training records."""

    final_dataset_path: Path
    final_manifest_path: Path
    augmentation_dataset_path: Path
    augmentation_manifest_path: Path
    output_dir: Path
    maximum_synthetic_fraction: float = 0.4

    def __post_init__(self) -> None:
        if not 0.0 <= self.maximum_synthetic_fraction < 1.0:
            raise ValueError("maximum_synthetic_fraction must be within [0, 1)")


def build_phase1_final_token_training_bundle(
    config: Phase1FinalTokenTrainingBundleConfig,
) -> dict[str, Any]:
    """Atomically materialize a source-pinned mixed-genre final-fit token dataset.

    INVARIANT: rows are copied verbatim. Their ``source_start``/``source_end`` offsets continue
    to address exactly the raw ``text`` field passed to the tokenizer.
    """

    final_summary = scan_span_dataset(config.final_dataset_path)
    augmentation_summary = scan_span_dataset(config.augmentation_dataset_path)
    final_manifest = validate_span_dataset_manifest(
        config.final_dataset_path,
        config.final_manifest_path,
        final_summary,
    )
    augmentation_manifest = validate_span_dataset_manifest(
        config.augmentation_dataset_path,
        config.augmentation_manifest_path,
        augmentation_summary,
    )
    _validate_source_manifests(final_manifest, augmentation_manifest)
    final_rows = read_jsonl(config.final_dataset_path)
    augmentation_rows = _approved_augmentation_rows(config.augmentation_dataset_path)
    _validate_unique_ids(final_rows, augmentation_rows)
    maximum = int(
        len(final_rows) * config.maximum_synthetic_fraction / (1.0 - config.maximum_synthetic_fraction)
    )
    selected_augmentation = tuple(sorted(augmentation_rows, key=_record_sort_key)[:maximum])
    if not selected_augmentation:
        raise ValueError("Final token training bundle found no approved Q&A/educational augmentation")
    combined = tuple(sorted((*final_rows, *selected_augmentation), key=_record_sort_key))
    output = config.output_dir
    build_contract = {
        "schema_version": _SCHEMA_VERSION,
        "final_dataset_sha256": final_summary.dataset_sha256,
        "final_manifest_sha256": sha256_file(config.final_manifest_path),
        "augmentation_dataset_sha256": augmentation_summary.dataset_sha256,
        "augmentation_manifest_sha256": sha256_file(config.augmentation_manifest_path),
        "synthetic_source_prefix": _SYNTHETIC_SOURCE_PREFIX,
        "maximum_synthetic_fraction": config.maximum_synthetic_fraction,
        "round2_included": False,
        "friend31_included": False,
    }
    build_key = sha256_text(json.dumps(build_contract, ensure_ascii=False, sort_keys=True))
    existing = _load_existing_bundle(output, build_key)
    if existing is not None:
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        dataset_path = staging / "spans.jsonl"
        manifest_path = staging / "manifest.json"
        dataset_sha256 = write_jsonl(dataset_path, combined)
        summary = scan_span_dataset(dataset_path)
        synthetic_fraction = len(selected_augmentation) / len(combined)
        if synthetic_fraction > config.maximum_synthetic_fraction:
            raise RuntimeError("Final token training bundle exceeded the synthetic fraction cap")
        entity_type_counts = Counter(
            str(entity["label"])
            for row in combined
            for entity in row["entities"]
        )
        manifest: dict[str, Any] = {
            "schema_version": "mined-span-dataset.v1",
            "document_count": len({str(row["document_id"]) for row in combined}),
            "chunk_count": summary.record_count,
            "empty_chunk_count": summary.empty_record_count,
            "entity_count": summary.entity_count,
            "split_chunk_counts": summary.split_record_counts,
            "entity_type_counts": dict(sorted(entity_type_counts.items())),
            "config": build_contract,
            "inputs": {
                "final_dataset": {
                    "path": str(config.final_dataset_path),
                    "sha256": final_summary.dataset_sha256,
                },
                "final_manifest": {
                    "path": str(config.final_manifest_path),
                    "sha256": sha256_file(config.final_manifest_path),
                },
                "augmentation_dataset": {
                    "path": str(config.augmentation_dataset_path),
                    "sha256": augmentation_summary.dataset_sha256,
                },
                "augmentation_manifest": {
                    "path": str(config.augmentation_manifest_path),
                    "sha256": sha256_file(config.augmentation_manifest_path),
                },
            },
            "output": "spans.jsonl",
            "output_sha256": dataset_sha256,
            "augmentation": {
                "selected_records": len(selected_augmentation),
                "selected_fraction": synthetic_fraction,
                "allowed_note_types": sorted(_ALLOWED_SYNTHETIC_NOTE_TYPES),
                "source_prefix": _SYNTHETIC_SOURCE_PREFIX,
            },
            "round2_included": False,
            "friend31_included": False,
        }
        write_json(manifest_path, manifest)
        # INVARIANT: validate the staged artifact before making it visible to a trainer.
        validate_span_dataset_manifest(dataset_path, manifest_path, summary)
        report = {
            "schema_version": _SCHEMA_VERSION,
            "build_key": build_key,
            "dataset": {
                "path": str(output / "spans.jsonl"),
                "sha256": dataset_sha256,
                "record_count": summary.record_count,
                "entity_count": summary.entity_count,
            },
            "augmentation": manifest["augmentation"],
            "round2_included": False,
            "friend31_included": False,
        }
        write_json(staging / "build_manifest.json", report)
        staging.replace(output)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_source_manifests(
    final_manifest: Mapping[str, Any],
    augmentation_manifest: Mapping[str, Any],
) -> None:
    for manifest in (final_manifest, augmentation_manifest):
        _reject_disallowed_provenance(manifest)


def _reject_disallowed_provenance(value: object, *, path: str = "manifest") -> None:
    """Reject affirmative Round 2, Friend31, or quarantine provenance anywhere in a manifest."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            child_path = f"{path}.{key}"
            if (
                any(marker in key_text for marker in ("round2", "friend31", "quarantined"))
                and child is True
            ):
                raise ValueError(f"Final token training bundle rejects disallowed provenance: {child_path}")
            _reject_disallowed_provenance(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_disallowed_provenance(child, path=f"{path}[{index}]")


def _approved_augmentation_rows(path: Path) -> tuple[dict[str, Any], ...]:
    """Select only renderer-generated train records, never re-copy manual records as synthetic."""

    approved: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        source = str(row.get("source_artifact_id", ""))
        note_type = str(row.get("note_type", ""))
        if source != _SYNTHETIC_SOURCE_PREFIX or note_type not in _ALLOWED_SYNTHETIC_NOTE_TYPES:
            continue
        if row.get("split") != "train":
            raise ValueError("Synthetic Q&A/educational augmentation must belong to the train split")
        copied = dict(row)
        copied["split"] = "train"
        approved.append(copied)
    return tuple(approved)


def _validate_unique_ids(
    final_rows: list[dict[str, Any]],
    augmentation_rows: tuple[dict[str, Any], ...],
) -> None:
    identifiers = [str(row.get("record_id", "")) for row in (*final_rows, *augmentation_rows)]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Final token training bundle requires globally unique record IDs")
    if any(
        not str(row.get("source_artifact_id", "")).startswith("phase1-final-supervision:")
        for row in final_rows
    ):
        raise ValueError("Final token training bundle final rows must be authorized supervision")


def _record_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["document_id"]), str(row["record_id"])


def _load_existing_bundle(output: Path, build_key: str) -> dict[str, Any] | None:
    if not output.exists():
        return None
    build_manifest = output / "build_manifest.json"
    if not build_manifest.is_file():
        raise FileExistsError(f"Training bundle exists without build manifest: {output}")
    payload = json.loads(build_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("build_key") != build_key:
        raise FileExistsError(f"Training bundle belongs to another build: {output}")
    dataset = output / "spans.jsonl"
    expected = payload.get("dataset", {}).get("sha256") if isinstance(payload.get("dataset"), dict) else None
    if not isinstance(expected, str) or not dataset.is_file() or sha256_file(dataset) != expected:
        raise ValueError("Existing final token training bundle failed its dataset fingerprint")
    return payload
