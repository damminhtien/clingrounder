"""Create deterministic, gold-blind review packs for neutral benchmark datasets.

The pack is deliberately separate from scoring.  A benchmark record may contain gold
annotations, but a reviewer must receive only the source text and a stable review ID.  The
coordinator keeps the ID map and later joins reviewed annotations back to the source dataset.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from clingrounder.evaluation.review_agreement import ReviewAgreementArtifact
from clingrounder.schema.types import RelationType

__all__ = [
    "ReviewPackConfig",
    "build_review_pack",
    "freeze_reviewed_snapshot",
    "import_review_pack",
]


_SCHEMA_VERSION = "clingrounder.review-pack.v1"
_ITEM_SCHEMA_VERSION = "clingrounder.review-pack-item.v1"
_IMPORT_SCHEMA_VERSION = "clingrounder.review-import.v1"
_SAFE_METADATA_KEYS = frozenset({"language", "genre", "note_type", "template_group"})


@dataclass(frozen=True, slots=True)
class ReviewPackConfig:
    """Deterministic reviewer assignment policy."""

    reviewers: tuple[str, ...] = ("reviewer-1", "reviewer-2")
    double_review_fraction: float = 0.10
    seed: int = 42

    def __post_init__(self) -> None:
        if len(self.reviewers) < 2:
            raise ValueError("A review pack requires at least two reviewers")
        if len(set(self.reviewers)) != len(self.reviewers):
            raise ValueError("Reviewers must be unique")
        if any(not reviewer.strip() for reviewer in self.reviewers):
            raise ValueError("Reviewer IDs must be non-empty")
        if not 0.0 <= self.double_review_fraction <= 1.0:
            raise ValueError("double_review_fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class _ReviewDocument:
    document_id: str
    text: str
    metadata: dict[str, str]


def build_review_pack(
    benchmark_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    config: ReviewPackConfig | None = None,
) -> dict[str, Any]:
    """Write a gold-blind review pack and return its deterministic manifest.

    ``output_dir`` is created if needed and must not be used as a staging directory for another
    pack.  Existing generated files are replaced atomically one at a time; unrelated files are
    left untouched so a coordinator can keep review notes beside the pack.
    """

    policy = config or ReviewPackConfig()
    benchmark_root = Path(benchmark_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    manifest_path = benchmark_root / "dataset_manifest.yaml"
    manifest = _load_manifest(manifest_path)
    dataset = _mapping(manifest.get("dataset"), "dataset")
    split_payload = _mapping(_mapping(manifest.get("splits"), "splits").get(split), split)
    input_path = (benchmark_root / str(split_payload["path"])).resolve()
    if benchmark_root not in input_path.parents:
        raise ValueError("Benchmark split path escapes the benchmark directory")

    documents = _load_documents(input_path)
    ordered = sorted(documents, key=lambda item: _assignment_key(item.document_id, policy.seed))
    double_count = math.ceil(len(ordered) * policy.double_review_fraction)
    double_reviewed = ordered[:double_count]
    single_reviewed = ordered[double_count:]
    assignments: dict[str, list[_ReviewDocument]] = {
        reviewer: list(double_reviewed) for reviewer in policy.reviewers
    }
    for index, document in enumerate(single_reviewed):
        assignments[policy.reviewers[index % len(policy.reviewers)]].append(document)

    output_root.mkdir(parents=True, exist_ok=True)
    for reviewer, reviewer_documents in assignments.items():
        reviewer_dir = output_root / reviewer
        reviewer_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            _review_item(dataset, split, document)
            for document in sorted(reviewer_documents, key=lambda item: item.document_id)
        ]
        _write_jsonl(reviewer_dir / "items.jsonl", rows)

    # INVARIANT: only the coordinator mapping contains source IDs. Reviewer files never include
    # gold fields or source IDs, so they can be handed to independent annotators safely.
    _write_jsonl(
        output_root / "coordinator_document_map.jsonl",
        (
            {
                "schema_version": _SCHEMA_VERSION,
                "review_id": _review_id(dataset, split, document.document_id),
                "document_id": document.document_id,
                "reviewers": [
                    reviewer
                    for reviewer, reviewer_documents in assignments.items()
                    if document in reviewer_documents
                ],
            }
            for document in sorted(documents, key=lambda item: item.document_id)
        ),
    )
    _write_readme(output_root, dataset, split, policy, len(documents), double_count)

    files = {
        str(path.relative_to(output_root)): _sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    result: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset": {
            "id": str(dataset.get("id", "")),
            "version": str(dataset.get("version", "")),
        },
        "split": split,
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_split_sha256": _sha256_file(input_path),
        "documents": len(documents),
        "reviewers": list(policy.reviewers),
        "double_review_fraction": policy.double_review_fraction,
        "double_reviewed_documents": double_count,
        "seed": policy.seed,
        "mutable_reviewer_files": [
            f"{reviewer}/items.jsonl" for reviewer in sorted(assignments)
        ],
        "assignments": {
            reviewer: len(values) for reviewer, values in sorted(assignments.items())
        },
        "files": files,
        "gold_blind": True,
    }
    _write_json(output_root / "manifest.json", result)
    return result


def import_review_pack(
    benchmark_dir: str | Path,
    pack_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    """Validate reviewer files and create an adjudication-ready local artifact.

    The importer deliberately stops before gold promotion.  It joins reviewer IDs back to source
    documents only after verifying the pack's source fingerprints and assignment map.  Agreement
    is reported when all submitted annotations are byte-equivalent; disagreement is preserved for
    a human adjudicator instead of being silently resolved.
    """

    benchmark_root = Path(benchmark_dir).expanduser().resolve()
    pack_root = Path(pack_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    source_manifest_path = benchmark_root / "dataset_manifest.yaml"
    source_manifest = _load_manifest(source_manifest_path)
    dataset = _mapping(source_manifest.get("dataset"), "dataset")
    split_payload = _mapping(_mapping(source_manifest.get("splits"), "splits").get(split), split)
    source_path = (benchmark_root / str(split_payload["path"])).resolve()
    if benchmark_root not in source_path.parents:
        raise ValueError("Benchmark split path escapes the benchmark directory")
    source_documents = {
        document.document_id: document for document in _load_documents(source_path)
    }

    pack_manifest_path = pack_root / "manifest.json"
    pack_manifest = _load_json_object(pack_manifest_path)
    if pack_manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"Unsupported review pack manifest: {pack_manifest_path}")
    if pack_manifest.get("gold_blind") is not True:
        raise ValueError("Review pack must be marked gold_blind")
    if pack_manifest.get("source_manifest_sha256") != _sha256_file(source_manifest_path):
        raise ValueError("Review pack source manifest fingerprint does not match benchmark")
    if pack_manifest.get("source_split_sha256") != _sha256_file(source_path):
        raise ValueError("Review pack source split fingerprint does not match benchmark")
    if pack_manifest.get("split") != split:
        raise ValueError(f"Review pack split mismatch: expected {split!r}")
    pack_dataset = _mapping(pack_manifest.get("dataset"), "review pack dataset")
    if (pack_dataset.get("id"), pack_dataset.get("version")) != (
        dataset.get("id"),
        dataset.get("version"),
    ):
        raise ValueError("Review pack dataset identity does not match benchmark")
    _verify_pack_files(pack_root, pack_manifest)

    mapping_path = pack_root / "coordinator_document_map.jsonl"
    mapping = _load_coordinator_mapping(mapping_path, source_documents)
    reviewers = _reviewers_from_manifest(pack_manifest)
    assignment_reviewers = {
        reviewer
        for assignment in mapping.values()
        for reviewer in assignment["reviewers"]
    }
    if not assignment_reviewers.issubset(reviewers):
        raise ValueError("Coordinator mapping contains an undeclared reviewer")
    if len(mapping) != len(source_documents):
        raise ValueError("Coordinator mapping does not cover the complete benchmark split")
    taxonomy = {
        field: _declared_values(source_manifest, field)
        for field in ("entities", "assertions", "code_systems")
    }
    submissions: list[dict[str, Any]] = []
    by_reviewer_and_id: dict[tuple[str, str], dict[str, Any]] = {}
    for reviewer in reviewers:
        reviewer_path = pack_root / reviewer / "items.jsonl"
        for item in _load_reviewer_items(reviewer_path):
            review_id = str(item["review_id"])
            if review_id not in mapping:
                raise ValueError(f"Reviewer {reviewer!r} submitted unknown review ID {review_id!r}")
            key = (reviewer, review_id)
            if key in by_reviewer_and_id:
                raise ValueError(f"Reviewer {reviewer!r} submitted duplicate review ID {review_id!r}")
            assignment = mapping[review_id]
            if reviewer not in assignment["reviewers"]:
                raise ValueError(f"Reviewer {reviewer!r} is not assigned to {review_id!r}")
            source_document = source_documents[assignment["document_id"]]
            if item["text"] != source_document.text:
                raise ValueError(f"Reviewer text mismatch for {review_id!r}")
            if dict(sorted(item["metadata"].items())) != dict(
                sorted(source_document.metadata.items())
            ):
                raise ValueError(f"Reviewer metadata mismatch for {review_id!r}")
            annotations = item["annotations"]
            relations = item["relations"]
            _validate_review_annotations(
                annotations,
                relations,
                source_document.text,
                taxonomy,
                review_id,
            )
            submission = {
                "schema_version": _IMPORT_SCHEMA_VERSION,
                "review_id": review_id,
                "document_id": source_document.document_id,
                "reviewer": reviewer,
                "text": source_document.text,
                "metadata": dict(sorted(source_document.metadata.items())),
                "entities": annotations,
                "relations": relations,
            }
            by_reviewer_and_id[key] = submission
            submissions.append(submission)

    _require_complete_assignments(mapping, by_reviewer_and_id)
    submissions.sort(key=lambda row: (row["review_id"], row["reviewer"]))
    adjudications = _build_adjudications(mapping, by_reviewer_and_id)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "submissions.jsonl", submissions)
    _write_jsonl(output_root / "adjudication.jsonl", adjudications)
    _write_import_readme(output_root)

    files = {
        str(path.relative_to(output_root)): _sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    result: dict[str, Any] = {
        "schema_version": _IMPORT_SCHEMA_VERSION,
        "dataset": {
            "id": str(dataset.get("id", "")),
            "version": str(dataset.get("version", "")),
        },
        "split": split,
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "source_split_sha256": _sha256_file(source_path),
        "review_pack_manifest_sha256": _sha256_file(pack_manifest_path),
        "reviewers": list(reviewers),
        "submission_count": len(submissions),
        "document_count": len(mapping),
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in adjudications).items())),
        "gold_promoted": False,
        "files": files,
    }
    _write_json(output_root / "manifest.json", result)
    return result


def freeze_reviewed_snapshot(
    benchmark_dir: str | Path,
    import_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    allow_single_review: bool = False,
) -> dict[str, Any]:
    """Freeze an explicitly completed review import as a separate dataset snapshot.

    The operation is deliberately separate from :func:`import_review_pack`.  Import only
    validates reviewer submissions and preserves disagreements; freeze requires every document
    to have either exact multi-reviewer agreement or an explicit adjudicator decision.  It never
    mutates the source benchmark manifest, so promotion remains a visible repository decision.

    INVARIANT: a blank or unfinished reviewer form cannot become a public gold snapshot.  The
    reviewer must set ``review_complete`` and an adjudicator must resolve every disagreement.
    """

    benchmark_root = Path(benchmark_dir).expanduser().resolve()
    import_root = Path(import_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    source_manifest_path = benchmark_root / "dataset_manifest.yaml"
    source_manifest = _load_manifest(source_manifest_path)
    dataset = _mapping(source_manifest.get("dataset"), "dataset")
    split_payload = _mapping(_mapping(source_manifest.get("splits"), "splits").get(split), split)
    source_path = (benchmark_root / str(split_payload["path"])).resolve()
    if benchmark_root not in source_path.parents:
        raise ValueError("Benchmark split path escapes the benchmark directory")
    source_documents = {
        document.document_id: document for document in _load_documents(source_path)
    }

    import_manifest_path = import_root / "manifest.json"
    import_manifest = _load_json_object(import_manifest_path)
    if import_manifest.get("schema_version") != _IMPORT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported review import manifest: {import_manifest_path}")
    if import_manifest.get("source_manifest_sha256") != _sha256_file(source_manifest_path):
        raise ValueError("Review import source manifest fingerprint does not match benchmark")
    if import_manifest.get("source_split_sha256") != _sha256_file(source_path):
        raise ValueError("Review import source split fingerprint does not match benchmark")
    if import_manifest.get("split") != split:
        raise ValueError(f"Review import split mismatch: expected {split!r}")

    adjudication_path = import_root / "adjudication.jsonl"
    adjudications = _load_adjudications(adjudication_path)
    taxonomy = {
        "entities": _declared_values(source_manifest, "entities"),
        "assertions": _declared_values(source_manifest, "assertions"),
        "code_systems": _declared_values(source_manifest, "code_systems"),
    }
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    seen_documents: set[str] = set()
    adjudicated_count = 0
    agreement_count = 0
    for row in adjudications:
        document_id = str(row.get("document_id", "")).strip()
        if document_id in seen_documents or document_id not in source_documents:
            raise ValueError(f"Review adjudication has invalid or duplicate document: {document_id!r}")
        seen_documents.add(document_id)
        status = row.get("status")
        status_counts[str(status)] += 1
        if status == "agreement":
            entities = row.get("agreed_entities")
            relations = row.get("agreed_relations")
            agreement_count += 1
        elif status == "adjudicated":
            entities = row.get("adjudicated_entities")
            relations = row.get("adjudicated_relations")
            adjudicated_count += 1
        elif status == "reviewed" and allow_single_review:
            submissions = row.get("submissions")
            if not isinstance(submissions, list) or len(submissions) != 1:
                raise ValueError(f"Single-review row has invalid submissions: {document_id!r}")
            entities = submissions[0].get("entities")
            relations = submissions[0].get("relations")
        else:
            raise ValueError(
                f"Review document {document_id!r} is not ready for snapshot: {status!r}"
            )
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise ValueError(f"Resolved annotations are missing for {document_id!r}")
        source_document = source_documents[document_id]
        _validate_review_annotations(
            entities,
            relations,
            source_document.text,
            taxonomy,
            f"snapshot:{document_id}",
        )
        rows.append(
            {
                "document_id": document_id,
                "text": source_document.text,
                "metadata": {
                    **dict(sorted(source_document.metadata.items())),
                    "human_reviewed": True,
                },
                "entities": entities,
                "relations": relations,
            }
        )

    if seen_documents != set(source_documents):
        missing = sorted(set(source_documents) - seen_documents)
        raise ValueError(f"Review adjudication is incomplete; missing documents: {missing}")
    if not rows:
        raise ValueError("Review adjudication is empty")

    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_root / f"{split}.jsonl"
    _write_jsonl(snapshot_path, sorted(rows, key=lambda row: row["document_id"]))
    double_reviewed_count = sum(
        1
        for row in adjudications
        if isinstance(row.get("submissions"), list) and len(row["submissions"]) > 1
    )
    if double_reviewed_count < 1:
        raise ValueError("A reviewed snapshot requires at least one double-reviewed document")
    agreement = ReviewAgreementArtifact(
        dataset_id=str(dataset.get("id", "")),
        dataset_version=f"{dataset.get('version', '')}-reviewed",
        reviewed_document_count=len(rows),
        double_reviewed_document_count=double_reviewed_count,
        double_review_fraction=double_reviewed_count / len(rows),
        span_type_agreement=_review_agreement_score(adjudications, _span_type_projection),
        assertion_agreement=_review_agreement_score(adjudications, _assertion_projection),
        relation_agreement=_review_agreement_score(adjudications, _relation_projection),
    )
    agreement_path = output_root / "review-agreement.json"
    _write_json(agreement_path, agreement.to_dict())
    reviewed_version = f"{dataset.get('version', '')}-reviewed"
    source_is_synthetic = _is_synthetic_dataset(dataset)
    snapshot_dataset_manifest = {
        "schema_version": "clingrounder.dataset-manifest.v1",
        "dataset": {
            "id": str(dataset.get("id", "")),
            "version": reviewed_version,
            # INVARIANT: reviewing a synthetic fixture improves annotation provenance but does
            # not turn it into clinical evidence. A real licensed source can be released.
            "status": "synthetic_reviewed" if source_is_synthetic else "released",
            "synthetic": source_is_synthetic,
            "language": dataset.get("language", ["und"]),
            "license": dataset.get("license", ""),
            "license_url": dataset.get("license_url", ""),
            "human_reviewed": True,
        },
        "splits": {
            split: {
                "path": snapshot_path.name,
                "documents": len(rows),
                "sha256": _sha256_file(snapshot_path),
            }
        },
        "entities": source_manifest.get("entities", []),
        "assertions": source_manifest.get("assertions", []),
        "code_systems": source_manifest.get("code_systems", []),
        "policy": {
            **dict(_mapping(source_manifest.get("policy"), "policy")),
            "test_used_for_development": False,
            "private_data": False,
        },
        "review": {
            "status": "released",
            "reviewers_required": max(2, len(import_manifest.get("reviewers", []))),
            "double_review_fraction": agreement.double_review_fraction,
            "agreement_targets": dict(
                _mapping(
                    _mapping(source_manifest.get("review"), "review").get(
                        "agreement_targets"
                    ),
                    "agreement_targets",
                )
            ),
            "agreement_report": agreement_path.name,
            "agreement_report_sha256": _sha256_file(agreement_path),
            "adjudication_required": True,
        },
    }
    dataset_manifest_path = output_root / "dataset_manifest.yaml"
    _write_yaml(dataset_manifest_path, snapshot_dataset_manifest)
    snapshot_manifest = {
        "schema_version": "clingrounder.reviewed-snapshot.v1",
        "dataset": {
            "id": str(dataset.get("id", "")),
            "version": reviewed_version,
        },
        "split": split,
        "human_reviewed": True,
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "source_split_sha256": _sha256_file(source_path),
        "review_import_manifest_sha256": _sha256_file(import_manifest_path),
        "snapshot_sha256": _sha256_file(snapshot_path),
        "agreement_sha256": _sha256_file(agreement_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "documents": len(rows),
        "agreement_count": agreement_count,
        "adjudicated_count": adjudicated_count,
        "status_counts": dict(sorted(status_counts.items())),
        "files": [
            snapshot_path.name,
            agreement_path.name,
            dataset_manifest_path.name,
        ],
    }
    _write_json(output_root / "manifest.json", snapshot_manifest)
    return snapshot_manifest


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one object manifest and reject arrays or scalar JSON values."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read JSON manifest: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must be an object: {path}")
    return payload


def _is_synthetic_dataset(dataset: Mapping[str, Any]) -> bool:
    """Recognize every declared synthetic lifecycle state conservatively.

    INVARIANT: Human review can improve synthetic annotation quality, but it cannot change the
    source class into clinical evidence. New synthetic states therefore inherit this boundary
    without requiring an exact status allowlist update.
    """

    status = str(dataset.get("status", "")).strip().casefold()
    return dataset.get("synthetic") is True or status.startswith("synthetic")


def _verify_pack_files(pack_root: Path, manifest: Mapping[str, Any]) -> None:
    """Verify every generated pack file before trusting reviewer assignments."""

    raw_files = manifest.get("files")
    if not isinstance(raw_files, Mapping):
        raise ValueError("Review pack manifest requires a files mapping")
    mutable_files = manifest.get("mutable_reviewer_files", [])
    if not isinstance(mutable_files, list) or not all(
        isinstance(value, str) for value in mutable_files
    ):
        raise ValueError("Review pack mutable_reviewer_files must be a string list")
    mutable_file_set = set(mutable_files)
    if any(
        relative not in raw_files or not relative.endswith("/items.jsonl")
        for relative in mutable_file_set
    ):
        raise ValueError("Review pack mutable files must be declared reviewer item files")
    for relative, expected_sha in raw_files.items():
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise ValueError("Review pack file manifest contains invalid entries")
        path = (pack_root / relative).resolve()
        if pack_root not in path.parents or not path.is_file():
            raise ValueError(f"Review pack file is missing or escapes the pack: {relative!r}")
        # INVARIANT: reviewer item files are intentionally editable after assignment. Their
        # content is validated below; immutable coordinator/provenance files remain hashed.
        if relative in mutable_file_set:
            continue
        if _sha256_file(path) != expected_sha:
            raise ValueError(f"Review pack file fingerprint mismatch: {relative!r}")


def _reviewers_from_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw_reviewers = manifest.get("reviewers")
    if not isinstance(raw_reviewers, list) or not raw_reviewers:
        raise ValueError("Review pack manifest requires reviewers")
    reviewers = tuple(str(value).strip() for value in raw_reviewers)
    if any(not value for value in reviewers) or len(set(reviewers)) != len(reviewers):
        raise ValueError("Review pack reviewers must be unique non-empty values")
    return reviewers


def _load_coordinator_mapping(
    path: Path,
    source_documents: Mapping[str, _ReviewDocument],
) -> dict[str, dict[str, Any]]:
    """Load the coordinator-only join map and verify every source document exists."""

    result: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping) or raw.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"Invalid coordinator mapping at line {line_number}")
        review_id = str(raw.get("review_id", "")).strip()
        document_id = str(raw.get("document_id", "")).strip()
        reviewers = raw.get("reviewers")
        if not review_id or not document_id or not isinstance(reviewers, list) or not reviewers:
            raise ValueError(f"Incomplete coordinator mapping at line {line_number}")
        reviewer_values = tuple(str(value).strip() for value in reviewers)
        if any(not value for value in reviewer_values) or len(set(reviewer_values)) != len(
            reviewer_values
        ):
            raise ValueError(f"Invalid coordinator reviewers at line {line_number}")
        if review_id in result:
            raise ValueError(f"Duplicate coordinator review ID {review_id!r}")
        if document_id not in source_documents:
            raise ValueError(f"Coordinator mapping references unknown document {document_id!r}")
        if any(item["document_id"] == document_id for item in result.values()):
            raise ValueError(f"Duplicate coordinator document ID {document_id!r}")
        result[review_id] = {
            "document_id": document_id,
            "reviewers": reviewer_values,
        }
    if not result:
        raise ValueError(f"Coordinator mapping is empty: {path}")
    return result


def _load_reviewer_items(path: Path) -> tuple[dict[str, Any], ...]:
    """Load reviewer JSONL while rejecting source IDs or hidden gold fields."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or raw.get("schema_version") != _ITEM_SCHEMA_VERSION:
            raise ValueError(f"Invalid reviewer item at {path}:{line_number}")
        if set(raw) != {
            "schema_version",
            "review_id",
            "text",
            "metadata",
            "annotations",
            "relations",
            "review_complete",
        }:
            raise ValueError(f"Reviewer item has forbidden fields at {path}:{line_number}")
        if not isinstance(raw.get("review_id"), str) or not raw["review_id"].strip():
            raise ValueError(f"Reviewer item requires review_id at {path}:{line_number}")
        if not isinstance(raw.get("text"), str) or not raw["text"]:
            raise ValueError(f"Reviewer item requires text at {path}:{line_number}")
        if not isinstance(raw.get("metadata"), Mapping):
            raise ValueError(f"Reviewer metadata must be an object at {path}:{line_number}")
        if not set(raw["metadata"]).issubset(_SAFE_METADATA_KEYS):
            raise ValueError(f"Reviewer metadata contains forbidden fields at {path}:{line_number}")
        if not all(isinstance(value, str) for value in raw["metadata"].values()):
            raise ValueError(f"Reviewer metadata values must be strings at {path}:{line_number}")
        if not isinstance(raw.get("annotations"), list) or not isinstance(raw.get("relations"), list):
            raise ValueError(f"Reviewer annotations/relations must be lists at {path}:{line_number}")
        if raw.get("review_complete") is not True:
            raise ValueError(
                f"Reviewer item must set review_complete=true at {path}:{line_number}"
            )
        rows.append(raw)
    return tuple(rows)


def _load_adjudications(path: Path) -> tuple[dict[str, Any], ...]:
    """Load adjudication rows while preserving explicit unresolved states."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Unable to read adjudication queue: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid adjudication JSON at {path}:{line_number}") from error
        if not isinstance(row, dict) or row.get("schema_version") != _IMPORT_SCHEMA_VERSION:
            raise ValueError(f"Invalid adjudication row at {path}:{line_number}")
        review_id = str(row.get("review_id", "")).strip()
        if not review_id or review_id in seen:
            raise ValueError(f"Duplicate/empty adjudication review ID at {path}:{line_number}")
        seen.add(review_id)
        rows.append(row)
    return tuple(rows)


def _review_agreement_score(
    adjudications: tuple[dict[str, Any], ...],
    projection: Any,
) -> float:
    """Measure exact inter-reviewer agreement for one annotation projection."""

    scores: list[float] = []
    for row in adjudications:
        submissions = row.get("submissions")
        if not isinstance(submissions, list) or len(submissions) < 2:
            continue
        projected = {
            json.dumps(projection(item), ensure_ascii=False, sort_keys=True)
            for item in submissions
        }
        scores.append(1.0 if len(projected) == 1 else 0.0)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _span_type_projection(submission: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                tuple(annotation.get("span", ())),
                annotation.get("type"),
            )
            for annotation in submission.get("entities", [])
        )
    )


def _assertion_projection(submission: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                tuple(annotation.get("span", ())),
                annotation.get("type"),
                annotation.get("assertion"),
            )
            for annotation in submission.get("entities", [])
        )
    )


def _relation_projection(submission: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                relation.get("head"),
                relation.get("tail"),
                relation.get("type"),
            )
            for relation in submission.get("relations", [])
        )
    )


def _require_complete_assignments(
    mapping: Mapping[str, Mapping[str, Any]],
    submissions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    """Fail closed when a reviewer omitted or added a document outside their assignment."""

    for review_id, assignment in mapping.items():
        for reviewer in assignment["reviewers"]:
            if (reviewer, review_id) not in submissions:
                raise ValueError(f"Missing reviewer submission for {reviewer!r}/{review_id!r}")


def _build_adjudications(
    mapping: Mapping[str, Mapping[str, Any]],
    submissions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group reviewer submissions while preserving disagreement for human adjudication."""

    result: list[dict[str, Any]] = []
    for review_id in sorted(mapping):
        assignment = mapping[review_id]
        ordered = [submissions[(reviewer, review_id)] for reviewer in assignment["reviewers"]]
        annotation_signatures = {
            _canonical_pair(item["entities"], item["relations"]) for item in ordered
        }
        agreed = len(annotation_signatures) == 1
        first = ordered[0]
        result.append(
            {
                "schema_version": _IMPORT_SCHEMA_VERSION,
                "review_id": review_id,
                "document_id": first["document_id"],
                "text": first["text"],
                "metadata": first["metadata"],
                "reviewers": [item["reviewer"] for item in ordered],
                "status": "agreement" if len(ordered) > 1 and agreed else (
                    "reviewed" if len(ordered) == 1 else "needs_adjudication"
                ),
                "submissions": [
                    {
                        "reviewer": item["reviewer"],
                        "entities": item["entities"],
                        "relations": item["relations"],
                    }
                    for item in ordered
                ],
                "agreed_entities": first["entities"] if agreed else None,
                "agreed_relations": first["relations"] if agreed else None,
            }
        )
    return result


def _canonical_pair(entities: object, relations: object) -> str:
    return json.dumps(
        {"entities": entities, "relations": relations},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_review_annotations(
    annotations: object,
    relations: object,
    source_text: str,
    taxonomy: Mapping[str, frozenset[str]],
    review_id: str,
) -> None:
    """Validate the neutral benchmark annotation contract before adjudication."""

    if not isinstance(annotations, list) or not isinstance(relations, list):
        raise ValueError(f"Review {review_id!r} annotations and relations must be lists")
    entity_ids: set[str] = set()
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError(f"Review {review_id!r} contains a non-object annotation")
        entity_id = str(annotation.get("id", "")).strip()
        if not entity_id or entity_id in entity_ids:
            raise ValueError(f"Review {review_id!r} contains duplicate/empty entity ID")
        entity_ids.add(entity_id)
        if annotation.get("type") not in taxonomy["entities"]:
            raise ValueError(f"Review {review_id!r} contains unsupported entity type")
        if annotation.get("assertion") not in taxonomy["assertions"]:
            raise ValueError(f"Review {review_id!r} contains unsupported assertion")
        code_system = annotation.get("code_system")
        code = annotation.get("code")
        if code_system not in taxonomy["code_systems"]:
            raise ValueError(f"Review {review_id!r} contains unsupported code system")
        if code_system == "NONE" and code is not None:
            raise ValueError(f"Review {review_id!r} assigns a code to NONE")
        if code_system != "NONE" and (not isinstance(code, str) or not code.strip()):
            raise ValueError(f"Review {review_id!r} omits a code for an assigned code system")
        span = annotation.get("span")
        if not isinstance(span, list | tuple) or len(span) != 2:
            raise ValueError(f"Review {review_id!r} contains an invalid span")
        start, end = span
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(source_text):
            raise ValueError(f"Review {review_id!r} contains an out-of-range span")
        if source_text[start:end] != annotation.get("text"):
            raise ValueError(f"Review {review_id!r} contains a span/text mismatch")
    relation_ids: set[str] = set()
    for relation in relations:
        if not isinstance(relation, Mapping):
            raise ValueError(f"Review {review_id!r} contains a non-object relation")
        relation_id = str(relation.get("id", "")).strip()
        head = str(relation.get("head", "")).strip()
        tail = str(relation.get("tail", "")).strip()
        if not relation_id or relation_id in relation_ids or head == tail:
            raise ValueError(f"Review {review_id!r} contains an invalid relation ID/endpoints")
        if head not in entity_ids or tail not in entity_ids:
            raise ValueError(f"Review {review_id!r} relation references an unknown entity")
        if not isinstance(relation.get("type"), str):
            raise ValueError(f"Review {review_id!r} relation type is required")
        try:
            RelationType(relation["type"])
        except ValueError as error:
            raise ValueError(f"Review {review_id!r} contains an unsupported relation type") from error
        relation_ids.add(relation_id)


def _write_import_readme(output_root: Path) -> None:
    """Document the deliberate boundary between agreement and gold promotion."""

    (output_root / "README.md").write_text(
        "# ClinGrounder review import\n\n"
        "This directory contains validated reviewer submissions and an adjudication queue.\n\n"
        "- `submissions.jsonl` preserves each reviewer's independent labels.\n"
        "- `adjudication.jsonl` marks single-review items, exact agreement, and disagreements.\n"
        "- `manifest.json` records source and pack fingerprints.\n\n"
        "Reviewer forms must set `review_complete: true` after annotation.\n\n"
        "`gold_promoted` is always false here. An adjudicator must resolve `needs_adjudication`,\n"
        "then run `review-snapshot-freeze` to explicitly export a reviewed snapshot and run the\n"
        "benchmark agreement gate.\n",
        encoding="utf-8",
    )


def _declared_values(manifest: Mapping[str, Any], field: str) -> frozenset[str]:
    """Read one non-empty benchmark taxonomy list for review validation."""

    values = manifest.get(field)
    if not isinstance(values, list) or not values or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError(f"Benchmark manifest requires a non-empty string list: {field}")
    return frozenset(value.strip() for value in values)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark manifest: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "clingrounder.dataset-manifest.v1":
        raise ValueError(f"Unsupported benchmark manifest: {path}")
    return payload


def _load_documents(path: Path) -> tuple[_ReviewDocument, ...]:
    documents: list[_ReviewDocument] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}:{line_number}: expected an object")
        document_id = str(raw.get("document_id", "")).strip()
        text = raw.get("text")
        if not document_id or not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number}: document_id and text are required")
        if document_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate document_id {document_id!r}")
        if not isinstance(raw.get("entities"), list) or not isinstance(raw.get("relations"), list):
            raise ValueError(f"{path}:{line_number}: gold entities and relations must be lists")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{path}:{line_number}: metadata must be an object")
        safe_metadata = {
            key: str(value)
            for key, value in metadata.items()
            if key in _SAFE_METADATA_KEYS and value is not None
        }
        seen.add(document_id)
        documents.append(_ReviewDocument(document_id, text, safe_metadata))
    return tuple(documents)


def _review_item(dataset: Mapping[str, Any], split: str, document: _ReviewDocument) -> dict[str, Any]:
    return {
        "schema_version": _ITEM_SCHEMA_VERSION,
        "review_id": _review_id(dataset, split, document.document_id),
        "text": document.text,
        "metadata": dict(sorted(document.metadata.items())),
        "annotations": [],
        "relations": [],
        "review_complete": False,
    }


def _review_id(dataset: Mapping[str, Any], split: str, document_id: str) -> str:
    identity = "\0".join((str(dataset.get("id", "")), str(dataset.get("version", "")), split, document_id))
    return f"review-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def _assignment_key(document_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{document_id}".encode("utf-8")).hexdigest()
    return digest, document_id


def _write_readme(
    output_root: Path,
    dataset: Mapping[str, Any],
    split: str,
    config: ReviewPackConfig,
    document_count: int,
    double_count: int,
) -> None:
    text = f"""# ClinGrounder review pack

This pack is a gold-blind annotation handoff for `{dataset.get('id', '')}` version
`{dataset.get('version', '')}`, split `{split}`.

- Documents: {document_count}
- Reviewers: {', '.join(config.reviewers)}
- Double-reviewed documents: {double_count}
- Assignment seed: {config.seed}

Reviewer files contain only `review_id`, source `text`, safe display metadata, empty
`annotations`/`relations` arrays, and `review_complete: false`. After annotation, reviewers must
set `review_complete: true`. Do not add source IDs, gold labels, or model predictions to the
reviewer handoff. The coordinator must retain `coordinator_document_map.jsonl` and use it to map
review IDs back to source document IDs after independent review.

Reviewer `items.jsonl` files are the only mutable files in the pack. Their initial hashes remain in
the pack manifest as acquisition provenance, but the importer validates the edited content rather
than requiring the pre-review hash.

Each annotation must preserve `[start, end)` offsets into the exact `text` field. Reviewers should
record unresolved decisions explicitly rather than guessing. This pack is a workflow artifact;
it does not make the underlying dataset human-reviewed or eligible for a clinical claim.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic dataset manifest without platform-specific line endings."""

    path.write_text(
        yaml.safe_dump(
            payload,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
