from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast


POLICY_HOLDOUT_SCHEMA_VERSION = "policy-holdout.v1"


def build_policy_holdout_manifest(
    documents_dir: str | Path,
    *,
    corpus_id: str,
    modulus: int = 5,
    holdout_bucket: int = 0,
) -> dict[str, Any]:
    """Seal a source-only split before holdout labels are created or inspected."""

    if not corpus_id.strip():
        raise ValueError("corpus_id must be non-empty")
    if modulus < 2 or not 0 <= holdout_bucket < modulus:
        raise ValueError("holdout_bucket must be within a modulus of at least two")
    document_paths = _document_paths(documents_dir)
    if not document_paths:
        raise ValueError("No source documents found")
    assignments = [
        {
            "document_id": document_id,
            "document_sha256": _file_sha256(path),
            "split": _split(document_id, modulus, holdout_bucket),
        }
        for document_id, path in document_paths.items()
    ]
    split_ids = {
        split: [row["document_id"] for row in assignments if row["split"] == split]
        for split in ("train", "holdout")
    }
    return {
        "schema_version": POLICY_HOLDOUT_SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "status": "sealed",
        "split_policy": {
            "algorithm": "int(sha256(document_id)[:8], 16) % modulus",
            "modulus": modulus,
            "holdout_bucket": holdout_bucket,
        },
        "corpus": {
            "document_count": len(assignments),
            "source_fingerprint_sha256": _canonical_sha256(assignments),
        },
        "splits": {
            split: {
                "document_count": len(split_ids[split]),
                "document_ids": split_ids[split],
            }
            for split in ("train", "holdout")
        },
        "assignments": assignments,
    }


def verify_policy_holdout_manifest(
    manifest: dict[str, Any],
    documents_dir: str | Path,
) -> None:
    """Verify source immutability without requiring or reading gold labels."""

    _validate_sealed_manifest(manifest)
    policy = manifest["split_policy"]
    current = build_policy_holdout_manifest(
        documents_dir,
        corpus_id=str(manifest["corpus_id"]),
        modulus=int(policy["modulus"]),
        holdout_bucket=int(policy["holdout_bucket"]),
    )
    if current != manifest:
        expected = manifest.get("corpus", {}).get("source_fingerprint_sha256")
        actual = current["corpus"]["source_fingerprint_sha256"]
        raise ValueError(
            "Sealed policy holdout no longer matches source documents: "
            f"expected {expected}, got {actual}."
        )


def open_policy_holdout_manifest(
    sealed_manifest: dict[str, Any],
    documents_dir: str | Path,
    holdout_gold_dir: str | Path,
) -> dict[str, Any]:
    """Create an opened record after policy selection is frozen.

    Gold must contain exactly holdout IDs. This prevents an all-corpus directory from being
    presented as a blind evaluation artifact while leaving the sealed manifest immutable.
    """

    verify_policy_holdout_manifest(sealed_manifest, documents_dir)
    expected_ids = set(sealed_manifest["splits"]["holdout"]["document_ids"])
    gold_paths = _gold_paths(holdout_gold_dir)
    actual_ids = set(gold_paths)
    if actual_ids != expected_ids:
        raise ValueError(
            "Holdout gold directory must contain exactly sealed holdout IDs: "
            f"missing={sorted(expected_ids - actual_ids, key=_sort_key)}, "
            f"unexpected={sorted(actual_ids - expected_ids, key=_sort_key)}."
        )
    assignments: list[dict[str, Any]] = []
    entity_count = 0
    for document_id, path in gold_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a JSON list")
        entity_count += len(payload)
        assignments.append(
            {
                "document_id": document_id,
                "gold_sha256": _file_sha256(path),
                "entity_count": len(payload),
            }
        )
    opened = cast(dict[str, Any], json.loads(json.dumps(sealed_manifest)))
    opened["status"] = "opened"
    opened["sealed_manifest_sha256"] = _canonical_sha256(sealed_manifest)
    opened["holdout_gold"] = {
        "document_count": len(assignments),
        "entity_count": entity_count,
        "fingerprint_sha256": _canonical_sha256(assignments),
        "assignments": assignments,
    }
    return opened


def write_policy_holdout_manifest(
    manifest: dict[str, Any],
    output_path: str | Path,
    *,
    replace: bool = False,
) -> None:
    output = Path(output_path)
    if output.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite policy holdout manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_sealed_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != POLICY_HOLDOUT_SCHEMA_VERSION:
        raise ValueError("Unsupported policy holdout manifest schema")
    if manifest.get("status") != "sealed":
        raise ValueError("Only a sealed manifest can be verified or opened")
    if "holdout_gold" in manifest:
        raise ValueError("Sealed manifest must not contain holdout labels")


def _document_paths(documents_dir: str | Path) -> dict[str, Path]:
    paths = {path.stem: path for path in Path(documents_dir).glob("*.txt")}
    return dict(sorted(paths.items(), key=lambda item: _sort_key(item[0])))


def _gold_paths(gold_dir: str | Path) -> dict[str, Path]:
    paths = {path.stem: path for path in Path(gold_dir).glob("*.json")}
    return dict(sorted(paths.items(), key=lambda item: _sort_key(item[0])))


def _split(document_id: str, modulus: int, holdout_bucket: int) -> str:
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8]
    return "holdout" if int(digest, 16) % modulus == holdout_bucket else "train"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)
