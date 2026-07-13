from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.evaluation.manual_gold_manifest import (
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
