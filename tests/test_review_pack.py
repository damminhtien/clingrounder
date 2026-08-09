"""Gold-blind benchmark review-pack contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clingrounder.evaluation.dataset_audit import audit_dataset
from clingrounder.evaluation.review_pack import (
    ReviewPackConfig,
    build_review_pack,
    freeze_reviewed_snapshot,
    import_review_pack,
)


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
                "review_complete",
                "review_id",
                "schema_version",
                "text",
            }
            assert row["annotations"] == []
            assert row["relations"] == []
            assert row["review_complete"] is False
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


def test_review_pack_import_validates_fingerprints_and_preserves_adjudication(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    build_review_pack(
        BENCHMARK,
        pack,
        config=ReviewPackConfig(reviewers=("alice", "bob"), double_review_fraction=0.5),
    )
    _complete_all_reviews(pack)

    result = import_review_pack(BENCHMARK, pack, tmp_path / "imported")

    assert result["gold_promoted"] is False
    assert result["submission_count"] == 6
    assert result["status_counts"] == {"agreement": 2, "reviewed": 2}
    assert (tmp_path / "imported" / "adjudication.jsonl").is_file()


def test_review_pack_import_allows_editable_items_and_marks_disagreement(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    build_review_pack(
        BENCHMARK,
        pack,
        config=ReviewPackConfig(reviewers=("alice", "bob"), double_review_fraction=1.0),
    )
    _complete_all_reviews(pack)
    alice_path = pack / "alice" / "items.jsonl"
    rows = [json.loads(line) for line in alice_path.read_text(encoding="utf-8").splitlines()]
    target = next(row for row in rows if "sốt" in row["text"])
    start = target["text"].index("sốt")
    target["annotations"] = [
        {
            "id": "review-e1",
            "span": [start, start + 3],
            "text": "sốt",
            "type": "SYMPTOM",
            "assertion": "NEGATED",
            "code_system": "LOCAL",
            "code": "SYMPTOM_FEVER",
        }
    ]
    alice_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = import_review_pack(BENCHMARK, pack, tmp_path / "imported")

    assert result["status_counts"] == {"agreement": 3, "needs_adjudication": 1}
    adjudications = [
        json.loads(line)
        for line in (tmp_path / "imported" / "adjudication.jsonl").read_text().splitlines()
    ]
    assert any(row["status"] == "needs_adjudication" for row in adjudications)


def test_review_pack_import_rejects_offset_mismatch(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    build_review_pack(BENCHMARK, pack, config=ReviewPackConfig(double_review_fraction=0.0))
    _complete_all_reviews(pack)
    reviewer_path = pack / "reviewer-1" / "items.jsonl"
    rows = [json.loads(line) for line in reviewer_path.read_text().splitlines()]
    rows[0]["annotations"] = [
        {
            "id": "bad",
            "span": [0, 1],
            "text": "wrong",
            "type": "SYMPTOM",
            "assertion": "PRESENT",
            "code_system": "NONE",
            "code": None,
        }
    ]
    reviewer_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    )

    with pytest.raises(ValueError, match="span/text mismatch"):
        import_review_pack(BENCHMARK, pack, tmp_path / "imported")


def test_review_pack_import_rejects_uncompleted_form(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    build_review_pack(BENCHMARK, pack, config=ReviewPackConfig(double_review_fraction=1.0))

    with pytest.raises(ValueError, match="review_complete=true"):
        import_review_pack(BENCHMARK, pack, tmp_path / "imported")


def test_reviewed_snapshot_requires_adjudication_for_disagreement(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    build_review_pack(
        BENCHMARK,
        pack,
        config=ReviewPackConfig(reviewers=("alice", "bob"), double_review_fraction=1.0),
    )
    _complete_all_reviews(pack)
    alice_path = pack / "alice" / "items.jsonl"
    rows = [json.loads(line) for line in alice_path.read_text(encoding="utf-8").splitlines()]
    target = next(row for row in rows if "sốt" in row["text"])
    start = target["text"].index("sốt")
    target["annotations"] = [
        {
            "id": "review-e1",
            "span": [start, start + 3],
            "text": "sốt",
            "type": "SYMPTOM",
            "assertion": "NEGATED",
            "code_system": "LOCAL",
            "code": "SYMPTOM_FEVER",
        }
    ]
    alice_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    imported = tmp_path / "imported"
    import_review_pack(BENCHMARK, pack, imported)

    with pytest.raises(ValueError, match="not ready for snapshot"):
        freeze_reviewed_snapshot(BENCHMARK, imported, tmp_path / "snapshot")


def test_reviewed_snapshot_requires_explicit_completion_and_writes_manifest(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    build_review_pack(
        BENCHMARK,
        pack,
        config=ReviewPackConfig(reviewers=("alice", "bob"), double_review_fraction=1.0),
    )
    _complete_all_reviews(pack)
    imported = tmp_path / "imported"
    import_review_pack(BENCHMARK, pack, imported)
    snapshot = tmp_path / "snapshot"

    result = freeze_reviewed_snapshot(BENCHMARK, imported, snapshot)

    assert result["schema_version"] == "clingrounder.reviewed-snapshot.v1"
    assert result["human_reviewed"] is True
    assert (snapshot / "test.jsonl").is_file()
    assert (snapshot / "review-agreement.json").is_file()
    assert (snapshot / "dataset_manifest.yaml").is_file()
    assert json.loads((snapshot / "manifest.json").read_text())["snapshot_sha256"]
    audit = audit_dataset(snapshot)
    assert audit.eligible_for_clinical_claim is False
    assert "clinical_claim_requires_human_review" in audit.warnings


def _complete_all_reviews(pack: Path) -> None:
    """Mark generated forms complete for tests that intentionally use empty labels."""

    for path in pack.glob("*/items.jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row["review_complete"] = True
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
