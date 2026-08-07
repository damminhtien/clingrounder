from __future__ import annotations

import json
from pathlib import Path

import pytest

from clingrounder.benchmarks.phase1.policy_holdout import (
    build_policy_holdout_manifest,
    open_policy_holdout_manifest,
    verify_policy_holdout_manifest,
)


def test_policy_holdout_is_sealed_before_gold_exists(tmp_path: Path) -> None:
    documents = _documents(tmp_path)

    manifest = build_policy_holdout_manifest(documents, corpus_id="next-task-v1")

    assert manifest["status"] == "sealed"
    assert "holdout_gold" not in manifest
    assert manifest["splits"]["train"]["document_count"] > 0
    assert manifest["splits"]["holdout"]["document_count"] > 0
    verify_policy_holdout_manifest(manifest, documents)


def test_policy_holdout_detects_source_changes(tmp_path: Path) -> None:
    documents = _documents(tmp_path)
    manifest = build_policy_holdout_manifest(documents, corpus_id="next-task-v1")
    (documents / "1.txt").write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="no longer matches"):
        verify_policy_holdout_manifest(manifest, documents)


def test_open_requires_exactly_holdout_gold_and_preserves_sealed_record(
    tmp_path: Path,
) -> None:
    documents = _documents(tmp_path)
    manifest = build_policy_holdout_manifest(documents, corpus_id="next-task-v1")
    original = json.loads(json.dumps(manifest))
    gold = tmp_path / "gold"
    gold.mkdir()
    for document_id in manifest["splits"]["holdout"]["document_ids"]:
        (gold / f"{document_id}.json").write_text("[]", encoding="utf-8")

    opened = open_policy_holdout_manifest(manifest, documents, gold)

    assert manifest == original
    assert opened["status"] == "opened"
    assert opened["holdout_gold"]["document_count"] == len(list(gold.glob("*.json")))
    assert opened["sealed_manifest_sha256"]

    train_id = manifest["splits"]["train"]["document_ids"][0]
    (gold / f"{train_id}.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly sealed holdout IDs"):
        open_policy_holdout_manifest(manifest, documents, gold)


def _documents(tmp_path: Path) -> Path:
    root = tmp_path / "documents"
    root.mkdir()
    for index in range(1, 21):
        (root / f"{index}.txt").write_text(f"document {index}", encoding="utf-8")
    return root
