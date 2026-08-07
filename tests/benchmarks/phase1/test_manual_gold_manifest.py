from __future__ import annotations

import json
from pathlib import Path

from clingrounder.benchmarks.phase1.manual_gold_manifest import (
    DEFAULT_CANDIDATE_POLICY,
    sync_manual_gold_manifest,
    validate_manual_gold_manifest,
    write_manual_gold_manifest,
)


def test_sync_manifest_preserves_review_notes_and_adds_missing_document(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    input_dir = tmp_path / "input"
    gold_dir.mkdir()
    input_dir.mkdir()
    (gold_dir / "1.json").write_text("[]", encoding="utf-8")
    (gold_dir / "2.json").write_text("[{}]", encoding="utf-8")
    (input_dir / "1.txt").write_text("", encoding="utf-8")
    (input_dir / "2.txt").write_text("", encoding="utf-8")
    manifest = gold_dir / "review_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "document_id": "1",
                "status": "reviewed",
                "reviewed_by": "reviewer",
                "review_date": "2026-07-01",
                "candidate_policy": "existing",
                "draft_policy": "existing",
                "entity_count": 9,
                "guideline_notes": ["keep this note"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = sync_manual_gold_manifest(
        gold_dir,
        input_dir,
        manifest,
        status="btc_v1_reviewed",
        reviewed_by="codex",
        review_date="2026-07-13",
    )

    assert [row["document_id"] for row in rows] == ["1", "2"]
    assert rows[0]["guideline_notes"] == ["keep this note"]
    assert rows[0]["entity_count"] == 0
    assert rows[1]["status"] == "btc_v1_reviewed"
    assert rows[1]["entity_count"] == 1
    assert write_manual_gold_manifest(rows, manifest) == 2
    assert validate_manual_gold_manifest({"1": [], "2": [{}]}, manifest) == []

    refreshed = sync_manual_gold_manifest(
        gold_dir,
        input_dir,
        manifest,
        status="btc_v1_reviewed",
        reviewed_by="codex",
        review_date="2026-07-13",
        refresh_candidate_policy=True,
    )
    assert {row["candidate_policy"] for row in refreshed} == {DEFAULT_CANDIDATE_POLICY}


def test_validate_manifest_reports_missing_and_stale_rows(tmp_path: Path) -> None:
    source = tmp_path / "1.txt"
    source.write_text("", encoding="utf-8")
    manifest = tmp_path / "review_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "document_id": "1",
                "status": "reviewed",
                "reviewed_by": "reviewer",
                "review_date": "2026-07-01",
                "candidate_policy": "exact",
                "draft_policy": "raw review",
                "entity_count": 0,
                "source_file": str(source),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_manual_gold_manifest({"1": [{}], "2": []}, manifest)

    assert {issue["kind"] for issue in issues} == {
        "missing_review_manifest_row",
        "review_manifest_entity_count",
    }


def test_sync_manifest_repairs_review_offsets_to_exact_raw_surface(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    input_dir = tmp_path / "input"
    gold_dir.mkdir()
    input_dir.mkdir()
    (gold_dir / "1.json").write_text("[]", encoding="utf-8")
    source_text = "Truyền  dịch"
    (input_dir / "1.txt").write_text(source_text, encoding="utf-8")
    manifest = gold_dir / "review_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "document_id": "1",
                "status": "reviewed",
                "reviewed_by": "reviewer",
                "review_date": "2026-07-14",
                "candidate_policy": "exact",
                "draft_policy": "raw review",
                "entity_count": 0,
                "review_candidates": [
                    {
                        "text": "truyền dịch",
                        "position": [1, 12],
                        "reason": "Procedure, not an entity.",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = sync_manual_gold_manifest(
        gold_dir,
        input_dir,
        manifest,
        status="reviewed",
        reviewed_by="reviewer",
        review_date="2026-07-14",
    )

    candidate = rows[0]["review_candidates"][0]
    assert candidate["text"] == "Truyền  dịch"
    assert candidate["position"] == [0, 12]
    write_manual_gold_manifest(rows, manifest)
    assert validate_manual_gold_manifest({"1": []}, manifest) == []


def test_sync_manifest_does_not_bind_short_alias_inside_a_word(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    input_dir = tmp_path / "input"
    gold_dir.mkdir()
    input_dir.mkdir()
    (gold_dir / "1.json").write_text("[]", encoding="utf-8")
    (input_dir / "1.txt").write_text("cho ho", encoding="utf-8")
    manifest = gold_dir / "review_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "document_id": "1",
                "status": "reviewed",
                "reviewed_by": "reviewer",
                "review_date": "2026-07-14",
                "candidate_policy": "exact",
                "draft_policy": "raw review",
                "entity_count": 0,
                "review_candidates": [
                    {
                        "text": "ho",
                        "position": [1, 3],
                        "reason": "Noisy token.",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = sync_manual_gold_manifest(
        gold_dir,
        input_dir,
        manifest,
        status="reviewed",
        reviewed_by="reviewer",
        review_date="2026-07-14",
    )

    assert rows[0]["review_candidates"][0]["position"] == [4, 6]
