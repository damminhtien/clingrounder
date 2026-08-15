"""Tests for the independent synthetic benchmark technical review."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_vi_clinical_benchmark import generate_snapshot
from scripts.review_vi_clinical_synthetic import review_synthetic_snapshot


def test_synthetic_technical_review_passes_without_clinical_eligibility(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    generate_snapshot(benchmark, train_documents=7, validation_documents=7, test_documents=14)

    report = review_synthetic_snapshot(benchmark, output_path=tmp_path / "review.json")

    assert report["status"] == "technical_review_pass"
    assert report["eligible_for_engineering_use"] is True
    assert report["eligible_for_clinical_claim"] is False
    assert report["clinical_claim_blockers"] == [
        "synthetic_source",
        "human_clinical_review_required",
    ]
    assert report["human_clinical_review"] is False
    assert report["documents"]["reviewed"] == 14
    assert report["documents"]["failed"] == 0
    assert json.loads((tmp_path / "review.json").read_text()) == report


def test_synthetic_technical_review_reports_offset_failure_without_text_leakage(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    generate_snapshot(benchmark, train_documents=1, validation_documents=1, test_documents=7)
    test_path = benchmark / "test.jsonl"
    row = json.loads(test_path.read_text(encoding="utf-8").splitlines()[0])
    row["entities"][0]["text"] = "tampered"
    test_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    report = review_synthetic_snapshot(benchmark)

    assert report["status"] == "technical_review_failed"
    assert report["eligible_for_engineering_use"] is False
    assert "offset_text_mismatch" in report["failures"][0]["checks"]
    rendered = json.dumps(report, ensure_ascii=False)
    assert "tampered" not in rendered
    assert "Bệnh nhân" not in rendered
