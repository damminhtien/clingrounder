"""Compile licensed mined spans into exact-quote instruction curricula.

The compiler is task-neutral: source labels are preserved exactly and no target-task
crosswalk is applied. A downstream specialization stage may learn a narrower schema,
but broad source semantics must not be silently rewritten during data preparation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.mining.io import write_json, write_jsonl
from clingrounder.mining.records import AccessClass
from clingrounder.mining.registry import load_source_registry
from clingrounder.utils.hashing import sha256_file, sha256_text
from clingrounder.utils.io import read_jsonl

__all__ = [
    "ExactQuoteCurriculumConfig",
    "build_exact_quote_curriculum",
]

_SCHEMA_VERSION = "exact-quote-curriculum.v1"
_SYSTEM_PROMPT = """Bạn nhận diện thực thể y khoa trong văn bản tiếng Việt.
Chỉ trả về JSON. Mỗi text phải là chuỗi trích nguyên văn, liên tục trong SOURCE.
Giữ nguyên nhãn nguồn được liệt kê trong TARGET_LABELS. Không đoán nhãn chi tiết hơn.
Không trả offset, assertion, mã bệnh hoặc mã thuốc."""


@dataclass(frozen=True, slots=True)
class ExactQuoteCurriculumConfig:
    """Immutable source, policy, and output identity for one curriculum."""

    source_id: str
    source_registry_path: Path
    spans_path: Path
    spans_manifest_path: Path
    output_dir: Path
    allowed_labels: tuple[str, ...]
    split: str = "train"

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("Curriculum source_id must be non-empty")
        if not self.source_registry_path.is_file():
            raise ValueError("Curriculum source registry does not exist")
        if not self.spans_path.is_file() or not self.spans_manifest_path.is_file():
            raise ValueError("Curriculum spans and manifest must exist")
        if not self.allowed_labels or self.allowed_labels != tuple(
            sorted(set(self.allowed_labels))
        ):
            raise ValueError("allowed_labels must be non-empty, unique, and sorted")
        if self.split != "train":
            raise ValueError("Training curriculum may read only the train split")


def build_exact_quote_curriculum(
    config: ExactQuoteCurriculumConfig,
) -> dict[str, Any]:
    """Build deterministic train-only instructions from a licensed span dataset."""

    source = load_source_registry(config.source_registry_path).by_id(config.source_id)
    if source.access_class is AccessClass.QUARANTINE:
        raise ValueError(f"Quarantined source cannot enter model training: {source.id}")
    if "entity_training" not in source.allowed_uses:
        raise ValueError(f"Source does not permit entity training: {source.id}")
    manifest = _load_span_manifest(
        config.spans_manifest_path,
        expected_sha256=sha256_file(config.spans_path),
    )
    rows = read_jsonl(config.spans_path)
    curriculum_rows = _build_rows(rows, config=config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "curriculum.jsonl"
    output_sha256 = write_jsonl(output_path, curriculum_rows)
    label_counts = Counter(
        entity["label"]
        for row in curriculum_rows
        for entity in row["targets"]
    )
    output_manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "source": {
            "id": source.id,
            "version": source.version,
            "license_id": source.license_id,
            "license_url": source.license_url,
            "access_class": source.access_class.value,
            "allowed_use": "entity_training",
            "hosted_processing_allowed": source.hosted_processing_allowed,
        },
        "inputs": {
            "registry": {
                "path": str(config.source_registry_path),
                "sha256": sha256_file(config.source_registry_path),
            },
            "spans": {
                "path": str(config.spans_path),
                "sha256": sha256_file(config.spans_path),
            },
            "spans_manifest": {
                "path": str(config.spans_manifest_path),
                "sha256": sha256_file(config.spans_manifest_path),
                "schema_version": manifest["schema_version"],
            },
        },
        "policy": {
            "split": config.split,
            "allowed_labels": list(config.allowed_labels),
            "source_labels_preserved": True,
            "target_task_crosswalk_applied": False,
            "offsets_in_assistant_output": False,
        },
        "output": {
            "path": output_path.name,
            "sha256": output_sha256,
            "record_count": len(curriculum_rows),
            "entity_count": sum(len(row["targets"]) for row in curriculum_rows),
            "label_counts": dict(sorted(label_counts.items())),
        },
    }
    output_manifest["build_fingerprint"] = sha256_text(
        json.dumps(output_manifest, ensure_ascii=False, sort_keys=True)
    )
    write_json(config.output_dir / "manifest.json", output_manifest)
    return output_manifest


def _load_span_manifest(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "mined-span-dataset.v1":
        raise ValueError("Unsupported mined span manifest")
    if raw.get("output_sha256") != expected_sha256:
        raise ValueError("Mined span dataset does not match its manifest")
    return raw


def _build_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: ExactQuoteCurriculumConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    observed_train = False
    for row in rows:
        split = str(row.get("split", ""))
        if split != config.split:
            continue
        observed_train = True
        text = str(row.get("text", ""))
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != row.get("text_sha256"):
            raise ValueError(f"Text fingerprint mismatch for {row.get('record_id')}")
        targets = _validated_targets(
            text,
            row.get("entities"),
            allowed_labels=config.allowed_labels,
            record_id=str(row.get("record_id", "")),
        )
        target_labels = ",".join(config.allowed_labels)
        user_prompt = (
            f"TARGET_LABELS: {target_labels}\n"
            "Trả JSON theo schema "
            '{"entities":[{"text":"exact quote","label":"SOURCE_LABEL"}]}.\n'
            f"SOURCE:\n{text}"
        )
        assistant = json.dumps(
            {"entities": targets},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        output.append(
            {
                "record_id": f"exact-quote:{config.source_id}:{row['record_id']}",
                "source_record_id": row["record_id"],
                "source_artifact_id": row["source_artifact_id"],
                "source_id": config.source_id,
                "split": config.split,
                "task": "vietnamese_biomedical_exact_quote",
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant},
                ],
                "targets": targets,
                "text_sha256": row["text_sha256"],
                "prompt_sha256": sha256_text(f"{_SYSTEM_PROMPT}\n{user_prompt}"),
            }
        )
    if not observed_train:
        raise ValueError("Curriculum source has no train records")
    return output


def _validated_targets(
    text: str,
    raw_entities: object,
    *,
    allowed_labels: tuple[str, ...],
    record_id: str,
) -> list[dict[str, str]]:
    if not isinstance(raw_entities, list):
        raise ValueError(f"Span entities must be a list for {record_id}")
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for raw in raw_entities:
        if not isinstance(raw, Mapping):
            raise ValueError(f"Span entity must be an object for {record_id}")
        label = str(raw.get("label", ""))
        if label not in allowed_labels:
            raise ValueError(f"Unexpected source label {label!r} for {record_id}")
        start = int(raw["start"])
        end = int(raw["end"])
        quote = str(raw.get("text", ""))
        # INVARIANT: model supervision is emitted only after validating immutable raw offsets.
        if start < 0 or end <= start or text[start:end] != quote:
            raise ValueError(f"Raw span mismatch for {record_id} at {(start, end)}")
        unique[(quote, label)] = {"text": quote, "label": label}
    return [
        unique[key]
        for key in sorted(unique, key=lambda item: (item[0], item[1]))
    ]
