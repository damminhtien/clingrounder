"""Materialize owner-authorized Phase 1 ground truth in its declared LF coordinate system."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from clingrounder.benchmarks.phase1.training_governance import (
    Phase1TrainingGovernance,
    load_phase1_training_governance,
)
from clingrounder.mining.io import write_json, write_jsonl
from clingrounder.benchmarks.phase1.ontology import PHASE1_ALLOWED_TYPES
from clingrounder.utils.hashing import sha256_file, sha256_text

__all__ = [
    "Phase1AuthorizedGroundTruthCorpus",
    "load_phase1_authorized_ground_truth",
    "materialize_phase1_authorized_ground_truth",
]

_SCHEMA_VERSION = "phase1-authorized-ground-truth.v1"
_SOURCE_PREFIX = "authorized_gt:"


@dataclass(frozen=True, slots=True)
class Phase1AuthorizedGroundTruthCorpus:
    """Immutable LF child documents plus exact Phase 1 labels from an authorized archive."""

    source_texts: Mapping[str, str]
    gold_rows: Mapping[str, tuple[Mapping[str, object], ...]]
    archive_sha256: str
    input_zip_sha256: str
    gt_zip_sha256: str
    offset_coordinate_view: str
    source_annotation_count: int
    duplicate_identity_count: int

    def __post_init__(self) -> None:
        if set(self.source_texts) != set(self.gold_rows):
            raise ValueError("Authorized ground-truth text and labels must share document IDs")
        if any(not document_id.startswith(_SOURCE_PREFIX) for document_id in self.source_texts):
            raise ValueError("Authorized ground-truth document IDs must use their source prefix")
        if self.offset_coordinate_view != "crlf_to_lf_child_document":
            raise ValueError("Authorized ground-truth must use its declared LF child view")


def load_phase1_authorized_ground_truth(
    governance_path: str | Path,
    *,
    archive_path: str | Path | None = None,
) -> Phase1AuthorizedGroundTruthCorpus:
    """Load one checksum-pinned archive without extracting private raw content to the repository.

    PRIVACY: the archive may be used for authorized supervised training, but it is never copied
    into a distributable data directory. Callers may materialize an encrypted/cache location with
    a manifest that contains checksums only.
    """

    governance = load_phase1_training_governance(governance_path)
    archive = _resolve_archive_path(governance, archive_path)
    policy = governance.authorized_ground_truth
    if sha256_file(archive) != policy.archive_sha256:
        raise ValueError("Authorized ground-truth archive SHA-256 does not match governance")
    input_payload, gt_payload = _read_nested_archives(archive)
    if _sha256_bytes(input_payload) != policy.input_zip_sha256:
        raise ValueError("Authorized ground-truth input ZIP SHA-256 does not match governance")
    if _sha256_bytes(gt_payload) != policy.gt_zip_sha256:
        raise ValueError("Authorized ground-truth annotation ZIP SHA-256 does not match governance")

    input_rows = _read_text_members(input_payload, expected_prefix="input/")
    annotation_rows = _read_annotation_members(gt_payload, expected_prefix="output/")
    if set(input_rows) != set(annotation_rows):
        raise ValueError("Authorized ground-truth input and annotation document IDs differ")
    if len(input_rows) != policy.expected_document_count:
        raise ValueError("Authorized ground-truth document count does not match governance")

    source_texts: dict[str, str] = {}
    gold_rows: dict[str, tuple[Mapping[str, object], ...]] = {}
    source_annotation_count = 0
    duplicate_identity_count = 0
    for original_id in sorted(input_rows, key=_document_sort_key):
        document_id = f"{_SOURCE_PREFIX}{original_id}"
        source_text = _to_lf_child_text(input_rows[original_id])
        source_rows = annotation_rows[original_id]
        rows, duplicate_count = _deduplicate_entity_identities(source_rows)
        source_annotation_count += len(source_rows)
        duplicate_identity_count += duplicate_count
        _validate_rows(document_id, source_text, rows)
        source_texts[document_id] = source_text
        gold_rows[document_id] = rows
    return Phase1AuthorizedGroundTruthCorpus(
        source_texts=source_texts,
        gold_rows=gold_rows,
        archive_sha256=policy.archive_sha256,
        input_zip_sha256=policy.input_zip_sha256,
        gt_zip_sha256=policy.gt_zip_sha256,
        offset_coordinate_view=policy.offset_coordinate_view,
        source_annotation_count=source_annotation_count,
        duplicate_identity_count=duplicate_identity_count,
    )


def materialize_phase1_authorized_ground_truth(
    corpus: Phase1AuthorizedGroundTruthCorpus,
    output_dir: str | Path,
) -> dict[str, object]:
    """Write a deterministic private training view with raw LF text and offset-validated labels."""

    target = Path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        documents = [
            {
                "document_id": document_id,
                "text": corpus.source_texts[document_id],
                "text_sha256": sha256_text(corpus.source_texts[document_id]),
                "source_id": "phase1_part2_leaked_bundle",
                "coordinate_view": corpus.offset_coordinate_view,
            }
            for document_id in sorted(corpus.source_texts, key=_document_sort_key)
        ]
        annotations = [
            {
                "document_id": document_id,
                "entities": list(corpus.gold_rows[document_id]),
            }
            for document_id in sorted(corpus.gold_rows, key=_document_sort_key)
        ]
        documents_sha256 = write_jsonl(staging / "documents.jsonl", documents)
        annotations_sha256 = write_jsonl(staging / "annotations.jsonl", annotations)
        manifest: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "source_id": "phase1_part2_leaked_bundle",
            "document_count": len(documents),
            "source_annotation_count": corpus.source_annotation_count,
            "deduplicated_annotation_count": sum(
                len(rows) for rows in corpus.gold_rows.values()
            ),
            "duplicate_identity_count": corpus.duplicate_identity_count,
            "offset_coordinate_view": corpus.offset_coordinate_view,
            "inputs": {
                "archive_sha256": corpus.archive_sha256,
                "input_zip_sha256": corpus.input_zip_sha256,
                "gt_zip_sha256": corpus.gt_zip_sha256,
            },
            "outputs": {
                "documents_sha256": documents_sha256,
                "annotations_sha256": annotations_sha256,
            },
        }
        write_json(staging / "manifest.json", manifest)
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _resolve_archive_path(
    governance: Phase1TrainingGovernance,
    archive_path: str | Path | None,
) -> Path:
    raw_path = archive_path or os.environ.get(governance.authorized_ground_truth.archive_env)
    if raw_path is None:
        raise ValueError(
            "Authorized ground-truth archive path is required via argument or configured environment"
        )
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_nested_archives(archive: Path) -> tuple[bytes, bytes]:
    try:
        with ZipFile(archive) as parent:
            names = set(parent.namelist())
            if {"input.zip", "gt.zip"} - names:
                raise ValueError("Authorized ground-truth archive has unexpected members")
            return parent.read("input.zip"), parent.read("gt.zip")
    except BadZipFile as error:
        raise ValueError("Authorized ground-truth archive is not a valid ZIP") from error


def _read_text_members(payload: bytes, *, expected_prefix: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                document_id = _member_document_id(info.filename, expected_prefix, ".txt")
                if document_id in rows:
                    raise ValueError("Authorized ground-truth contains duplicate document text")
                rows[document_id] = archive.read(info).decode("utf-8")
    except BadZipFile as error:
        raise ValueError("Authorized ground-truth input member is not a valid ZIP") from error
    return rows


def _read_annotation_members(
    payload: bytes,
    *,
    expected_prefix: str,
) -> dict[str, tuple[Mapping[str, object], ...]]:
    rows: dict[str, tuple[Mapping[str, object], ...]] = {}
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                document_id = _member_document_id(info.filename, expected_prefix, ".json")
                if document_id in rows:
                    raise ValueError("Authorized ground-truth contains duplicate annotations")
                parsed = json.loads(archive.read(info).decode("utf-8"))
                if not isinstance(parsed, list) or not all(isinstance(item, Mapping) for item in parsed):
                    raise ValueError("Authorized ground-truth annotation member must be an entity list")
                rows[document_id] = tuple(dict(item) for item in parsed)
    except BadZipFile as error:
        raise ValueError("Authorized ground-truth annotation member is not a valid ZIP") from error
    return rows


def _member_document_id(name: str, prefix: str, suffix: str) -> str:
    if not name.startswith(prefix) or "/" in name[len(prefix) :] or not name.endswith(suffix):
        raise ValueError("Authorized ground-truth ZIP contains an unexpected member path")
    document_id = name[len(prefix) : -len(suffix)]
    if not document_id or not document_id.isdecimal():
        raise ValueError("Authorized ground-truth document ID must be numeric")
    return document_id


def _validate_rows(
    document_id: str,
    source_text: str,
    rows: tuple[Mapping[str, object], ...],
) -> None:
    for row in rows:
        text = row.get("text")
        entity_type = row.get("type")
        position = row.get("position")
        if not isinstance(text, str) or entity_type not in PHASE1_ALLOWED_TYPES:
            raise ValueError(f"{document_id}: annotation has invalid text/type")
        if not isinstance(position, list) or len(position) != 2 or not all(isinstance(value, int) and not isinstance(value, bool) for value in position):
            raise ValueError(f"{document_id}: annotation has invalid position")
        start, end = position
        if (
            start < 0
            or end <= start
            or end > len(source_text)
            or source_text[start:end] != text
        ):
            raise ValueError(f"{document_id}: annotation violates the LF raw-offset invariant")


def _deduplicate_entity_identities(
    rows: tuple[Mapping[str, object], ...],
) -> tuple[tuple[Mapping[str, object], ...], int]:
    """Collapse duplicate exact NER targets while preserving their source count in the manifest."""

    deduplicated: list[Mapping[str, object]] = []
    seen: set[tuple[object, object, tuple[object, ...]]] = set()
    duplicates = 0
    for row in rows:
        position = row.get("position")
        key = (
            row.get("text"),
            row.get("type"),
            tuple(position) if isinstance(position, list) else (),
        )
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduplicated.append(row)
    return tuple(deduplicated), duplicates


def _to_lf_child_text(source_text: str) -> str:
    """Convert only line endings; any other text change would invalidate organizer offsets."""

    return source_text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _document_sort_key(document_id: str) -> tuple[int, str]:
    raw = document_id.removeprefix(_SOURCE_PREFIX)
    return (int(raw) if raw.isdecimal() else -1, document_id)
