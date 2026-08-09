"""Gold-blind benchmark review-pack contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clingrounder.evaluation.review_pack import ReviewPackConfig, build_review_pack


BENCHMARK = Path("benchmarks/vi_clinical_grounding_v1")


def test_review_pack_is_deterministic_and_does_not_export_gold(tmp_path: Path) -> None:
    first = build_review_pack(
        BENCHMARK,
        tmp_path / "first",
        split="test",
        config=ReviewPackConfig(
            reviewers=("alice", "bob"),
            double_review_fraction=0.5,
            seed=7,
        ),
    )
    second = build_review_pack(
        BENCHMARK,
        tmp_path / "second",
        split="test",
        config=ReviewPackConfig(
            reviewers=("alice", "bob"),
            double_review_fraction=0.5,
            seed=7,
        ),
    )

    assert first == second
    assert first["gold_blind"] is True
    assert first["double_reviewed_documents"] == 2
    assert first["assignments"] == {"alice": 3, "bob": 3}

    for reviewer in ("alice", "bob"):
        path = tmp_path / "first" / reviewer / "items.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows
        for row in rows:
            assert set(row) == {
                "annotations",
                "metadata",
                "relations",
                "review_id",
                "schema_version",
                "text",
            }
            assert row["annotations"] == []
            assert row["relations"] == []
            assert "entities" not in row
            assert "document_id" not in row


def test_review_pack_keeps_coordinator_mapping_out_of_reviewer_payload(tmp_path: Path) -> None:
    build_review_pack(BENCHMARK, tmp_path / "pack", config=ReviewPackConfig(double_review_fraction=0.0))

    mapping = tmp_path / "pack" / "coordinator_document_map.jsonl"
    rows = [json.loads(line) for line in mapping.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert all(
        set(row) == {"document_id", "review_id", "reviewers", "schema_version"}
        for row in rows
    )
    assert all(len(row["reviewers"]) == 1 for row in rows)


def test_review_pack_rejects_invalid_assignment_policy() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ReviewPackConfig(reviewers=("only",))
    with pytest.raises(ValueError, match="unique"):
        ReviewPackConfig(reviewers=("same", "same"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ReviewPackConfig(reviewers=("a", "b"), double_review_fraction=1.1)
