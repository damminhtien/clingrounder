"""Leakage-safe instruction and hard-negative datasets for Qwen Phase 1 runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.qwen_proposals import (
    Phase1AdjudicationCandidate,
    Phase1AdjudicationDecision,
    Phase1ReviewEntity,
    build_phase1_qwen_adjudication_messages,
    build_phase1_qwen_extraction_messages,
    build_phase1_qwen_review_messages,
)
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.io import read_jsonl

__all__ = [
    "Phase1QwenDatasetConfig",
    "build_phase1_qwen_instruction_dataset",
]

_SCHEMA_VERSION = "phase1-qwen-instructions.v2"
_FORBIDDEN_SOURCE_MARKERS = (
    "round2",
    "round-2",
    "input_part2",
    "leak",
    "quarantine",
)
_LABEL_MAP = {
    "SYMPTOM": "TRIỆU_CHỨNG",
    "LAB_TEST": "TÊN_XÉT_NGHIỆM",
    "LAB_RESULT": "KẾT_QUẢ_XÉT_NGHIỆM",
    "DISEASE": "CHẨN_ĐOÁN",
    "DRUG": "THUỐC",
}


@dataclass(frozen=True, slots=True)
class Phase1QwenDatasetConfig:
    """Immutable inputs for extraction, review, and optional adjudication curricula."""

    spans_path: Path
    spans_manifest_path: Path
    output_dir: Path
    hard_negative_predictions_path: Path | None = None
    include_development: bool = True
    review_masks_per_train_record: int = 0
    review_keep_fraction: float = 0.5
    review_seed: str = "phase1-qwen-review-missing-v1"

    def __post_init__(self) -> None:
        if not self.spans_path.is_file() or not self.spans_manifest_path.is_file():
            raise ValueError("Qwen dataset source spans and manifest must exist")
        if (
            self.hard_negative_predictions_path is not None
            and not self.hard_negative_predictions_path.is_file()
        ):
            raise ValueError("Hard-negative prediction file does not exist")
        if not 0 <= self.review_masks_per_train_record <= 4:
            raise ValueError("Review masks per train record must be between zero and four")
        if not 0.0 <= self.review_keep_fraction < 1.0:
            raise ValueError("Review keep fraction must be in [0, 1)")
        if not self.review_seed.strip():
            raise ValueError("Review seed must be non-empty")


def build_phase1_qwen_instruction_dataset(
    config: Phase1QwenDatasetConfig,
) -> dict[str, Any]:
    """Build deterministic extraction, missing-review, and adjudication records."""

    source_manifest = _load_and_validate_source_manifest(
        config.spans_manifest_path,
        spans_sha256=sha256_file(config.spans_path),
    )
    source_rows = read_jsonl(config.spans_path)
    _validate_source_rows(source_rows)
    extraction_rows = [
        _extraction_record(row)
        for row in source_rows
        if row["split"] == "train"
        or (config.include_development and row["split"] == "development")
    ]
    hard_negative_rows: list[dict[str, Any]] = []
    if config.hard_negative_predictions_path is not None:
        hard_negative_rows = _build_hard_negative_records(
            source_rows,
            read_jsonl(config.hard_negative_predictions_path),
        )
    review_rows = _build_review_records(source_rows, config=config)

    target = config.output_dir
    target.mkdir(parents=True, exist_ok=True)
    extraction_path = target / "extraction.jsonl"
    hard_negative_path = target / "hard_negatives.jsonl"
    review_path = target / "review_missing.jsonl"
    extraction_sha256 = write_jsonl(extraction_path, extraction_rows)
    hard_negative_sha256 = write_jsonl(hard_negative_path, hard_negative_rows)
    review_sha256 = write_jsonl(review_path, review_rows)
    split_counts = Counter(str(row["split"]) for row in extraction_rows)
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "inputs": {
            "spans": {
                "path": str(config.spans_path),
                "sha256": sha256_file(config.spans_path),
            },
            "spans_manifest": {
                "path": str(config.spans_manifest_path),
                "sha256": sha256_file(config.spans_manifest_path),
            },
            "hard_negative_predictions": (
                None
                if config.hard_negative_predictions_path is None
                else {
                    "path": str(config.hard_negative_predictions_path),
                    "sha256": sha256_file(config.hard_negative_predictions_path),
                }
            ),
        },
        "source_contract": {
            "round2_included": False,
            "quarantined_data_included": False,
            "source_schema_version": source_manifest["schema_version"],
        },
        "outputs": {
            "extraction": {
                "path": extraction_path.name,
                "sha256": extraction_sha256,
                "record_count": len(extraction_rows),
                "split_counts": dict(sorted(split_counts.items())),
            },
            "hard_negatives": {
                "path": hard_negative_path.name,
                "sha256": hard_negative_sha256,
                "record_count": len(hard_negative_rows),
            },
            "review_missing": {
                "path": review_path.name,
                "sha256": review_sha256,
                "record_count": len(review_rows),
                "train_only": True,
                "masks_per_train_record": config.review_masks_per_train_record,
                "keep_fraction": config.review_keep_fraction,
                "seed": config.review_seed,
            },
        },
    }
    manifest["build_fingerprint"] = _mapping_sha256(
        {
            "schema_version": _SCHEMA_VERSION,
            "inputs": manifest["inputs"],
            "outputs": manifest["outputs"],
        }
    )
    write_json(target / "manifest.json", manifest)
    return manifest


def _load_and_validate_source_manifest(
    path: Path,
    *,
    spans_sha256: str,
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Span dataset manifest must be an object")
    if raw.get("schema_version") != "mined-span-dataset.v1":
        raise ValueError("Unsupported span dataset schema")
    if raw.get("output_sha256") != spans_sha256:
        raise ValueError("Span dataset fingerprint does not match its manifest")
    augmentation = raw.get("augmentation", {})
    if not isinstance(augmentation, Mapping):
        augmentation = {}
    if bool(augmentation.get("round2_included")):
        raise ValueError("Round 2 data cannot enter Qwen training or calibration")
    if bool(augmentation.get("quarantined_data_included")):
        raise ValueError("Quarantined data cannot enter Qwen training or calibration")
    synthetic_fraction = float(augmentation.get("synthetic_train_fraction", 0.0))
    if synthetic_fraction > 0.4:
        raise ValueError("Qwen synthetic training fraction cannot exceed 0.4")
    if int(augmentation.get("synthetic_development_record_count", 0)) != 0:
        raise ValueError("Synthetic records cannot enter Qwen development")
    return raw


def _validate_source_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Qwen instruction source dataset is empty")
    observed_splits: set[str] = set()
    for row in rows:
        split = str(row.get("split", ""))
        observed_splits.add(split)
        if split not in {"train", "development"}:
            raise ValueError(f"Forbidden Qwen supervision split: {split!r}")
        identity = " ".join(
            str(row.get(field, "")).lower()
            for field in ("document_id", "source_artifact_id", "record_id")
        )
        if any(marker in identity for marker in _FORBIDDEN_SOURCE_MARKERS):
            raise ValueError(f"Forbidden Qwen supervision source: {identity}")
        text = str(row.get("text", ""))
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != row.get("text_sha256"):
            raise ValueError(f"Text fingerprint mismatch for {row.get('record_id')}")
        entities = row.get("entities")
        if not isinstance(entities, list):
            raise ValueError("Span record entities must be a list")
        for entity in entities:
            if not isinstance(entity, Mapping):
                raise ValueError("Span entity must be an object")
            start = int(entity["start"])
            end = int(entity["end"])
            if text[start:end] != entity.get("text"):
                raise ValueError(
                    f"Raw span mismatch for {row.get('record_id')} at {(start, end)}"
                )
            if entity.get("label") not in _LABEL_MAP:
                raise ValueError(f"Unsupported Qwen entity label: {entity.get('label')}")
    if observed_splits != {"train", "development"}:
        raise ValueError("Qwen source requires non-empty train and development splits")


def _extraction_record(row: Mapping[str, Any]) -> dict[str, Any]:
    text = str(row["text"])
    target_types = tuple(sorted(_LABEL_MAP.values()))
    messages = build_phase1_qwen_extraction_messages(
        text,
        pass_id="supervised-recall",
        target_types=target_types,  # type: ignore[arg-type]
    )
    unique_targets: dict[tuple[str, str], dict[str, Any]] = {}
    for entity in row["entities"]:
        key = (str(entity["text"]), _LABEL_MAP[str(entity["label"])])
        unique_targets[key] = {
            "text": key[0],
            "type": key[1],
        }
    assistant = json.dumps(
        {
            "entities": [
                unique_targets[key]
                for key in sorted(unique_targets, key=lambda item: (item[0], item[1]))
            ]
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "record_id": f"qwen-extract:{row['record_id']}",
        "source_record_id": row["record_id"],
        "source_artifact_id": row["source_artifact_id"],
        "document_id": row["document_id"],
        "split": row["split"],
        "task": "phase1_entity_extraction",
        "messages": [
            {"role": message.role, "content": message.content} for message in messages
        ]
        + [{"role": "assistant", "content": assistant}],
        "text_sha256": row["text_sha256"],
        "entity_count": len(unique_targets),
    }


def _build_review_records(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    config: Phase1QwenDatasetConfig,
) -> list[dict[str, Any]]:
    """Create deterministic missing-only masks from train records.

    MODEL: the reviewer learns to complement an existing extractor, matching the iterative
    inference protocol. Development rows are never masked into extra training examples.
    """

    output: list[dict[str, Any]] = []
    if config.review_masks_per_train_record == 0:
        return output
    for row in source_rows:
        if row["split"] != "train":
            continue
        unique = _unique_review_entities(row)
        if not unique:
            continue
        for mask_index in range(config.review_masks_per_train_record):
            kept, missing = _partition_review_entities(
                unique,
                record_id=str(row["record_id"]),
                mask_index=mask_index,
                keep_fraction=config.review_keep_fraction,
                seed=config.review_seed,
            )
            messages = build_phase1_qwen_review_messages(
                str(row["text"]),
                tuple(
                    Phase1ReviewEntity(
                        text=str(entity["text"]),
                        entity_type=str(entity["type"]),  # type: ignore[arg-type]
                        span=(int(entity["start"]), int(entity["end"])),
                    )
                    for entity in kept
                ),
                round_index=mask_index,
            )
            assistant = json.dumps(
                {
                    "entities": [
                        {
                            "text": str(entity["text"]),
                            "type": str(entity["type"]),
                        }
                        for entity in missing
                    ]
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            output.append(
                {
                    "record_id": (
                        f"qwen-review:{row['record_id']}:mask-{mask_index:02d}"
                    ),
                    "source_record_id": row["record_id"],
                    "source_artifact_id": row["source_artifact_id"],
                    "document_id": row["document_id"],
                    "split": "train",
                    "task": "phase1_missing_entity_review",
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ]
                    + [{"role": "assistant", "content": assistant}],
                    "text_sha256": row["text_sha256"],
                    "existing_entity_count": len(kept),
                    "missing_entity_count": len(missing),
                    "mask_index": mask_index,
                }
            )
    return output


def _unique_review_entities(
    row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    text = str(row["text"])
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for entity in row["entities"]:
        source_label = str(entity["label"])
        phase1_label = _LABEL_MAP[source_label]
        start = int(entity["start"])
        end = int(entity["end"])
        quote = text[start:end]
        key = (quote, phase1_label)
        current = unique.get(key)
        candidate = {
            "text": quote,
            "type": phase1_label,
            "start": start,
            "end": end,
        }
        if current is None or (start, end) < (
            int(current["start"]),
            int(current["end"]),
        ):
            unique[key] = candidate
    return sorted(
        unique.values(),
        key=lambda entity: (
            int(entity["start"]),
            int(entity["end"]),
            str(entity["type"]),
        ),
    )


def _partition_review_entities(
    entities: Sequence[dict[str, Any]],
    *,
    record_id: str,
    mask_index: int,
    keep_fraction: float,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    ranked = sorted(
        entities,
        key=lambda entity: hashlib.sha256(
            (
                f"{seed}\0{record_id}\0{mask_index}\0"
                f"{entity['text']}\0{entity['type']}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    threshold = int(keep_fraction * (1 << 64))
    for entity in ranked:
        digest = hashlib.sha256(
            (
                f"{seed}\0keep\0{record_id}\0{mask_index}\0"
                f"{entity['text']}\0{entity['type']}"
            ).encode("utf-8")
        ).hexdigest()
        destination = kept if int(digest[:16], 16) < threshold else missing
        destination.append(entity)
    if not missing:
        missing.append(kept.pop())
    if len(entities) > 1 and keep_fraction > 0.0 and not kept:
        kept.append(missing.pop(0))

    def entity_key(entity: Mapping[str, Any]) -> tuple[int, int, str]:
        return (
            int(entity["start"]),
            int(entity["end"]),
            str(entity["type"]),
        )

    return sorted(kept, key=entity_key), sorted(missing, key=entity_key)


def _build_hard_negative_records(
    source_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Turn train-only XLM-R errors into DROP/REPLACE adjudication examples."""

    train_by_source_id = {
        _source_document_id(str(row["document_id"])): row
        for row in source_rows
        if row["split"] == "train"
        and str(row["source_artifact_id"]).startswith("phase1-manual-gold:")
    }
    output: list[dict[str, Any]] = []
    for prediction in predictions:
        document_id = str(prediction.get("document_id", ""))
        source = train_by_source_id.get(document_id)
        if source is None:
            raise ValueError(
                "Hard-negative predictions must contain train documents only; "
                f"unmatched document_id={document_id!r}"
            )
        text = str(source["text"])
        gold = [
            (
                int(entity["start"]),
                int(entity["end"]),
                _LABEL_MAP[str(entity["label"])],
            )
            for entity in source["entities"]
        ]
        for index, entity in enumerate(prediction.get("entities", [])):
            span = tuple(int(value) for value in entity["span"])
            predicted_type = _LABEL_MAP.get(str(entity["type"]))
            if predicted_type is None or len(span) != 2:
                continue
            start, end = span
            if start < 0 or end <= start or text[start:end] != entity.get("text"):
                raise ValueError("Hard-negative prediction violates source offsets")
            exact = (start, end, predicted_type) in gold
            if exact:
                continue
            overlapping_gold = [
                item for item in gold if item[0] < end and start < item[1]
            ]
            candidate = Phase1AdjudicationCandidate(
                proposal_id=f"xlmr-{index}",
                text=text[start:end],
                entity_type=predicted_type,  # type: ignore[arg-type]
                span=(start, end),
                sources=("xlmr",),
                confidence=float(entity.get("confidence", 0.0)),
            )
            if (
                len(overlapping_gold) == 1
                and overlapping_gold[0][2] == predicted_type
            ):
                gold_start, gold_end, _ = overlapping_gold[0]
                decision = Phase1AdjudicationDecision(
                    proposal_id=candidate.proposal_id,
                    action="REPLACE_BOUNDARY",
                    replacement_text=text[gold_start:gold_end],
                )
            elif len(overlapping_gold) == 1 and (
                overlapping_gold[0][0],
                overlapping_gold[0][1],
            ) == (start, end):
                decision = Phase1AdjudicationDecision(
                    proposal_id=candidate.proposal_id,
                    action="REPLACE_TYPE",
                    replacement_type=overlapping_gold[0][2],  # type: ignore[arg-type]
                )
            else:
                decision = Phase1AdjudicationDecision(
                    proposal_id=candidate.proposal_id,
                    action="DROP",
                )
            messages = build_phase1_qwen_adjudication_messages(text, (candidate,))
            assistant = json.dumps(
                {
                    "decisions": [
                        {
                            "proposal_id": decision.proposal_id,
                            "action": decision.action,
                            "replacement_text": decision.replacement_text,
                            "replacement_type": decision.replacement_type,
                        }
                    ]
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            output.append(
                {
                    "record_id": (
                        f"qwen-hard-negative:{source['record_id']}:{candidate.proposal_id}"
                    ),
                    "source_record_id": source["record_id"],
                    "source_artifact_id": source["source_artifact_id"],
                    "document_id": source["document_id"],
                    "split": "train",
                    "task": "phase1_entity_adjudication",
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ]
                    + [{"role": "assistant", "content": assistant}],
                    "text_sha256": source["text_sha256"],
                    "decision": decision.action,
                }
            )
    return output


def _source_document_id(document_id: str) -> str:
    return document_id.rsplit(":", 1)[-1]


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
