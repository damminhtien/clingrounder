"""Create deterministic, gold-blind review packs for neutral benchmark datasets.

The pack is deliberately separate from scoring.  A benchmark record may contain gold
annotations, but a reviewer must receive only the source text and a stable review ID.  The
coordinator keeps the ID map and later joins reviewed annotations back to the source dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = ["ReviewPackConfig", "build_review_pack"]


_SCHEMA_VERSION = "clingrounder.review-pack.v1"
_ITEM_SCHEMA_VERSION = "clingrounder.review-pack-item.v1"
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
        "assignments": {
            reviewer: len(values) for reviewer, values in sorted(assignments.items())
        },
        "files": files,
        "gold_blind": True,
    }
    _write_json(output_root / "manifest.json", result)
    return result


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

Reviewer files contain only `review_id`, source `text`, safe display metadata, and empty
`annotations`/`relations` arrays. Do not add source IDs, gold labels, or model predictions to the
reviewer handoff. The coordinator must retain `coordinator_document_map.jsonl` and use it to map
review IDs back to source document IDs after independent review.

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
