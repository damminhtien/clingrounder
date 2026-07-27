"""Import user-provided Phase 1 synthetic data into a leakage-safe NER view.

The source archive contains contest-shaped labels, but it is not clinician-reviewed gold. This
module therefore validates every split while exporting only a bounded subset of ``train``. The
human development split remains unchanged and candidate/assertion labels are deliberately omitted
from the model-neutral span dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.training.span_dataset import (
    SpanTrainingRecord,
    scan_span_dataset,
    validate_span_dataset_manifest,
)
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text
from medical_kg_nlp.utils.io import read_jsonl

__all__ = [
    "Phase1SyntheticTrainingConfig",
    "build_phase1_synthetic_training_dataset",
]

_SCHEMA_VERSION = "phase1-user-synthetic-training.v1"
_SOURCE_SCHEMA_VERSION = "viettel-medical-synthetic-v1"
_ALLOWED_SPLITS = ("train", "dev", "test")
_ALLOWED_TYPES = {
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
}
_ALLOWED_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}
_LABEL_MAP = {
    "TRIỆU_CHỨNG": "SYMPTOM",
    "TÊN_XÉT_NGHIỆM": "LAB_TEST",
    "KẾT_QUẢ_XÉT_NGHIỆM": "LAB_RESULT",
    "CHẨN_ĐOÁN": "DISEASE",
    "THUỐC": "DRUG",
}
_STYLE_ORDER = (
    "qa_advice",
    "hybrid_mixed",
    "noisy_translation",
    "terse_ehr",
    "case_report",
    "structured_discharge",
)


@dataclass(frozen=True, slots=True)
class Phase1SyntheticTrainingConfig:
    """Pinned archive and human dataset inputs for one deterministic build."""

    archive_path: Path
    expected_archive_sha256: str
    human_spans_path: Path
    human_manifest_path: Path
    output_dir: Path
    max_synthetic_train_fraction: float = 0.4
    selection_seed: str = "phase1-user-synthetic-balanced-v1"
    source_id: str = "viettel-medical-synthetic-v1"

    def __post_init__(self) -> None:
        if not self.archive_path.is_file():
            raise ValueError("Synthetic source archive does not exist")
        if not self.human_spans_path.is_file() or not self.human_manifest_path.is_file():
            raise ValueError("Human span dataset and manifest must exist")
        if not re.fullmatch(r"[0-9a-f]{64}", self.expected_archive_sha256):
            raise ValueError("Expected synthetic archive SHA-256 must be lowercase hex")
        if not 0.0 <= self.max_synthetic_train_fraction <= 0.4:
            raise ValueError("Synthetic train fraction must be in [0, 0.4]")
        if not self.selection_seed.strip() or not self.source_id.strip():
            raise ValueError("Synthetic selection seed and source ID must be non-empty")


def build_phase1_synthetic_training_dataset(
    config: Phase1SyntheticTrainingConfig,
) -> dict[str, Any]:
    """Build an atomic span dataset with synthetic train rows and human-only development."""

    archive_sha256 = sha256_file(config.archive_path)
    if archive_sha256 != config.expected_archive_sha256:
        raise ValueError(
            "Synthetic archive fingerprint mismatch: "
            f"expected {config.expected_archive_sha256}, observed {archive_sha256}"
        )
    human_rows = _load_human_rows(config)
    source_rows, source_audit = _load_and_audit_archive(config.archive_path)
    human_train_count = sum(row["split"] == "train" for row in human_rows)
    maximum_synthetic = _maximum_synthetic_records(
        human_train_count,
        config.max_synthetic_train_fraction,
    )
    selected = _select_balanced_train_rows(
        source_rows["train"],
        maximum=maximum_synthetic,
        seed=config.selection_seed,
        blocked_texts={_normalized_text(str(row["text"])) for row in human_rows},
    )
    synthetic_rows = [
        _convert_synthetic_row(
            row,
            archive_sha256=archive_sha256,
            source_id=config.source_id,
        )
        for row in selected
    ]
    output_rows = [*human_rows, *synthetic_rows]
    synthetic_fraction = len(synthetic_rows) / max(
        1,
        human_train_count + len(synthetic_rows),
    )
    if synthetic_fraction > config.max_synthetic_train_fraction + 1e-12:
        raise RuntimeError("Synthetic selection exceeded the configured training fraction")

    build_contract = {
        "schema_version": _SCHEMA_VERSION,
        "archive_sha256": archive_sha256,
        "human_spans_sha256": sha256_file(config.human_spans_path),
        "human_manifest_sha256": sha256_file(config.human_manifest_path),
        "maximum_synthetic_train_fraction": config.max_synthetic_train_fraction,
        "selection_seed": config.selection_seed,
        "source_id": config.source_id,
        "builder_sha256": sha256_file(Path(__file__)),
    }
    build_key = _mapping_sha256(build_contract)
    existing = _load_existing_build(config.output_dir, build_key)
    if existing is not None:
        return existing

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_dir.name}.",
            dir=config.output_dir.parent,
        )
    )
    try:
        spans_path = staging / "spans.jsonl"
        output_sha256 = write_jsonl(spans_path, output_rows)
        summary = scan_span_dataset(spans_path)
        type_counts = Counter(
            str(entity["label"])
            for row in output_rows
            for entity in _entity_rows(row)
        )
        manifest = {
            "schema_version": "mined-span-dataset.v1",
            "document_count": len({str(row["document_id"]) for row in output_rows}),
            "chunk_count": summary.record_count,
            "empty_chunk_count": summary.empty_record_count,
            "entity_count": summary.entity_count,
            "entity_type_counts": dict(sorted(type_counts.items())),
            "split_chunk_counts": dict(summary.split_record_counts),
            "config": {
                "max_synthetic_train_fraction": config.max_synthetic_train_fraction,
                "selection_policy": "balanced_style_round_robin",
                "selection_seed": config.selection_seed,
                "style_order": list(_STYLE_ORDER),
            },
            "inputs": {
                "archive": {
                    "path": str(config.archive_path),
                    "sha256": archive_sha256,
                    "schema_version": _SOURCE_SCHEMA_VERSION,
                },
                "human_spans": _fingerprinted_path(config.human_spans_path),
                "human_manifest": _fingerprinted_path(config.human_manifest_path),
            },
            "output": spans_path.name,
            "output_sha256": output_sha256,
            "augmentation": {
                "source_train_record_count": human_train_count,
                "source_development_record_count": sum(
                    row["split"] == "development" for row in human_rows
                ),
                "synthetic_train_record_count": len(synthetic_rows),
                "synthetic_train_fraction": synthetic_fraction,
                "synthetic_development_record_count": 0,
                "development_augmented": False,
                "round2_included": False,
                "quarantined_data_included": False,
                "evaluation_eligible": False,
                "candidate_labels_exported": False,
                "assertion_labels_exported": False,
                "label_origin": (
                    "synthetic_templates_with_codes_derived_from_user_reference_ground_truth"
                ),
            },
        }
        manifest_path = staging / "manifest.json"
        write_json(manifest_path, manifest)
        validate_span_dataset_manifest(spans_path, manifest_path, summary)
        write_json(staging / "source_audit.json", source_audit)
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
                "synthetic_train_fraction": synthetic_fraction,
                "selected_style_counts": dict(
                    sorted(Counter(str(row["style"]) for row in selected).items())
                ),
            },
            "outputs": {
                "manifest.json": sha256_file(manifest_path),
                "source_audit.json": sha256_file(staging / "source_audit.json"),
                "spans.jsonl": sha256_file(spans_path),
            },
        }
        write_json(staging / "build_manifest.json", build_manifest)
        staging.replace(config.output_dir)
        return build_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_human_rows(
    config: Phase1SyntheticTrainingConfig,
) -> list[dict[str, Any]]:
    summary = scan_span_dataset(config.human_spans_path)
    validate_span_dataset_manifest(
        config.human_spans_path,
        config.human_manifest_path,
        summary,
    )
    rows = read_jsonl(config.human_spans_path)
    observed_splits = {str(row.get("split", "")) for row in rows}
    if observed_splits != {"train", "development"}:
        raise ValueError("Human source requires train and development splits only")
    for row in rows:
        SpanTrainingRecord.from_mapping(row)
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("origin") == "synthetic":
            raise ValueError("Human source already contains synthetic rows")
    return rows


def _load_and_audit_archive(
    archive_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    split_rows: dict[str, list[dict[str, Any]]] = {}
    record_ids: set[str] = set()
    text_hashes: dict[str, str] = {}
    entity_counts: Counter[str] = Counter()
    assertion_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    with ZipFile(archive_path) as archive:
        for split in _ALLOWED_SPLITS:
            member = _unique_member(archive, f"jsonl/{split}.jsonl")
            rows: list[dict[str, Any]] = []
            for line_number, line in enumerate(
                archive.read(member).decode("utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ValueError(f"{member}:{line_number}: row must be an object")
                row = {str(key): value for key, value in raw.items()}
                _validate_source_row(row, split=split, location=f"{member}:{line_number}")
                record_id = str(row["id"])
                if record_id in record_ids:
                    raise ValueError(f"Duplicate synthetic record ID {record_id!r}")
                record_ids.add(record_id)
                normalized_hash = sha256_text(_normalized_text(str(row["text"])))
                previous_split = text_hashes.get(normalized_hash)
                if previous_split is not None:
                    raise ValueError(
                        "Normalized synthetic document crossed source splits: "
                        f"{record_id!r} and {previous_split!r}"
                    )
                text_hashes[normalized_hash] = f"{split}:{record_id}"
                style_counts[f"{split}:{row['style']}"] += 1
                for entity in _entity_rows(row):
                    entity_counts[f"{split}:{entity['type']}"] += 1
                    assertion_counts.update(str(value) for value in entity["assertions"])
                rows.append(row)
            if not rows:
                raise ValueError(f"Synthetic source split {split!r} is empty")
            split_rows[split] = rows
    return split_rows, {
        "schema_version": "phase1-user-synthetic-source-audit.v1",
        "archive_sha256": sha256_file(archive_path),
        "record_counts": {
            split: len(rows) for split, rows in sorted(split_rows.items())
        },
        "entity_counts": dict(sorted(entity_counts.items())),
        "assertion_counts": dict(sorted(assertion_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
        "offset_issue_count": 0,
        "overlap_issue_count": 0,
        "cross_split_normalized_duplicate_count": 0,
        "training_policy": {
            "exported_source_split": "train",
            "source_dev_exported": False,
            "source_test_exported": False,
            "candidate_labels_exported": False,
            "assertion_labels_exported": False,
        },
    }


def _validate_source_row(
    row: Mapping[str, Any],
    *,
    split: str,
    location: str,
) -> None:
    record_id = row.get("id")
    text = row.get("text")
    style = row.get("style")
    entities = row.get("entities")
    if not isinstance(record_id, str) or not record_id.startswith(f"{split}-"):
        raise ValueError(f"{location}: invalid split-scoped record ID")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{location}: text must be non-empty")
    if not isinstance(style, str) or not style:
        raise ValueError(f"{location}: style must be non-empty")
    if not isinstance(entities, list):
        raise ValueError(f"{location}: entities must be a list")

    ordered: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            raise ValueError(f"{location}: entity {index} must be an object")
        entity_type = entity.get("type")
        position = entity.get("position")
        entity_text = entity.get("text")
        assertions = entity.get("assertions")
        candidates = entity.get("candidates")
        if entity_type not in _ALLOWED_TYPES:
            raise ValueError(f"{location}: entity {index} has an invalid type")
        if (
            not isinstance(position, list)
            or len(position) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in position)
        ):
            raise ValueError(f"{location}: entity {index} has an invalid position")
        start, end = int(position[0]), int(position[1])
        if start < 0 or end <= start or end > len(text) or text[start:end] != entity_text:
            raise ValueError(f"{location}: entity {index} violates raw offsets")
        if not isinstance(assertions, list) or not set(assertions) <= _ALLOWED_ASSERTIONS:
            raise ValueError(f"{location}: entity {index} has invalid assertions")
        if not isinstance(candidates, list) or not all(
            isinstance(value, str) for value in candidates
        ):
            raise ValueError(f"{location}: entity {index} has invalid candidates")
        if entity_type not in {"CHẨN_ĐOÁN", "THUỐC"} and candidates:
            raise ValueError(f"{location}: entity {index} has candidates on an invalid type")
        identity = (start, end, str(entity_type))
        if identity in seen:
            raise ValueError(f"{location}: duplicate entity identity {identity}")
        seen.add(identity)
        ordered.append(identity)
    ordered.sort()
    for left, right in zip(ordered, ordered[1:], strict=False):
        if right[0] < left[1]:
            raise ValueError(f"{location}: overlapping entities are not BIO-compatible")


def _select_balanced_train_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum: int,
    seed: str,
    blocked_texts: set[str],
) -> list[dict[str, Any]]:
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = {str(key): value for key, value in raw.items()}
        if _normalized_text(str(row["text"])) in blocked_texts:
            continue
        by_style[str(row["style"])].append(row)
    for style_rows in by_style.values():
        style_rows.sort(
            key=lambda row: (
                hashlib.sha256(f"{seed}\0{row['id']}".encode("utf-8")).hexdigest(),
                str(row["id"]),
            )
        )

    style_order = [style for style in _STYLE_ORDER if by_style.get(style)]
    style_order.extend(sorted(set(by_style) - set(style_order)))
    selected: list[dict[str, Any]] = []
    cursor: Counter[str] = Counter()
    while len(selected) < maximum:
        added = False
        for style in style_order:
            index = cursor[style]
            if index >= len(by_style[style]):
                continue
            selected.append(by_style[style][index])
            cursor[style] += 1
            added = True
            if len(selected) == maximum:
                break
        if not added:
            break
    return selected


def _convert_synthetic_row(
    row: Mapping[str, Any],
    *,
    archive_sha256: str,
    source_id: str,
) -> dict[str, Any]:
    text = str(row["text"])
    source_record_id = str(row["id"])
    digest = sha256_text(f"{archive_sha256}\0{source_record_id}\0{text}")
    document_id = f"phase1-user-synthetic:{digest[:24]}"
    entities = []
    for index, entity in enumerate(_entity_rows(row)):
        start, end = (int(value) for value in entity["position"])
        entities.append(
            {
                "annotation_id": f"phase1-user-synthetic-ann:{digest[:20]}:{index:03d}",
                "start": start,
                "end": end,
                "source_start": start,
                "source_end": end,
                "text": str(entity["text"]),
                "label": _LABEL_MAP[str(entity["type"])],
            }
        )
    output = {
        "record_id": f"phase1-user-synthetic-record:{digest[:24]}",
        "document_id": document_id,
        "split": "train",
        "text": text,
        "text_sha256": sha256_text(text),
        "source_span": [0, len(text)],
        "language": "vi",
        "note_type": str(row["style"]),
        "source_artifact_id": f"user-synthetic:{source_id}@{archive_sha256}",
        "entities": entities,
        "metadata": {
            "origin": "synthetic",
            "evaluation_eligible": False,
            "source_record_id": source_record_id,
            "source_split": "train",
            "source_archive_sha256": archive_sha256,
            "candidate_labels_exported": False,
            "assertion_labels_exported": False,
        },
    }
    # INVARIANT: the converted text owns its coordinate space; no normalized offset is emitted.
    SpanTrainingRecord.from_mapping(output)
    return output


def _unique_member(archive: ZipFile, suffix: str) -> str:
    matches = []
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member {name!r}")
        if name == suffix or name.endswith(f"/{suffix}"):
            matches.append(name)
    if len(matches) != 1:
        raise ValueError(
            f"Synthetic archive requires exactly one {suffix!r}; observed {matches}"
        )
    return matches[0]


def _maximum_synthetic_records(
    human_train_count: int,
    maximum_fraction: float,
) -> int:
    if human_train_count <= 0 or maximum_fraction <= 0.0:
        return 0
    return math.floor(
        maximum_fraction * human_train_count / (1.0 - maximum_fraction)
    )


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _entity_rows(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entities = row.get("entities")
    if not isinstance(entities, list):
        raise ValueError("Span row entities must be a list")
    if not all(isinstance(entity, Mapping) for entity in entities):
        raise ValueError("Span row entities must contain objects")
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


def _load_existing_build(target: Path, build_key: str) -> dict[str, Any] | None:
    manifest_path = target / "build_manifest.json"
    if not manifest_path.is_file():
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite non-versioned dataset directory {target}"
            )
        return None
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("build_key") != build_key:
        raise FileExistsError(
            f"Dataset directory {target} belongs to a different build"
        )
    return raw
