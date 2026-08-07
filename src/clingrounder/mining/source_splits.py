"""Freeze source-declared dataset splits without reassigning documents."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from clingrounder.mining.io import write_json
from clingrounder.mining.records import MinedDocument
from clingrounder.utils.hashing import sha256_file

__all__ = ["freeze_source_splits"]


def freeze_source_splits(
    documents: Sequence[MinedDocument],
    *,
    metadata_key: str,
    split_map: Mapping[str, str],
    documents_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Persist a source split mapping after checking coverage and target names."""

    if not metadata_key.strip():
        raise ValueError("Source split metadata_key must be non-empty")
    if not split_map or any(
        not source.strip() or not target.strip()
        for source, target in split_map.items()
    ):
        raise ValueError("Source split_map must contain non-empty names")
    splits: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for document in sorted(documents, key=lambda item: item.document_id):
        source_split = document.metadata.get(metadata_key, "")
        target_split = split_map.get(source_split)
        if target_split is None:
            raise ValueError(
                f"Unmapped source split {source_split!r} for {document.document_id}"
            )
        if document.document_id in splits:
            raise ValueError(f"Duplicate document ID {document.document_id!r}")
        splits[document.document_id] = target_split
        counts[target_split] += 1
    if not splits:
        raise ValueError("Cannot freeze source splits for an empty dataset")
    manifest = {
        "schema_version": "source-split-manifest.v1",
        "inputs": {
            "documents": {
                "path": str(documents_path),
                "sha256": sha256_file(documents_path),
            }
        },
        "metadata_key": metadata_key,
        "split_map": dict(sorted(split_map.items())),
        "split_counts": dict(sorted(counts.items())),
        "splits": splits,
    }
    write_json(output_path, manifest)
    return manifest
