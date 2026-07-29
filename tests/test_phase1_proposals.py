from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.phase1_proposals import (
    build_phase1_proposal_matrix,
    proposal_consensus_keys,
    write_phase1_proposal_matrix,
)


def test_proposal_matrix_classifies_exact_overlap_type_conflict_and_source_only() -> None:
    text = "đau ngực và ho"
    sources = {
        "codex": {"1": [_row("đau ngực", "TRIỆU_CHỨNG", 0), _row("ho", "TRIỆU_CHỨNG", 12)]},
        "pipeline": {
            "1": [
                _row("đau ngực", "TRIỆU_CHỨNG", 0),
                _row("đau ngực", "CHẨN_ĐOÁN", 0),
                _row("ngực", "TRIỆU_CHỨNG", 4),
            ]
        },
        "qwen": {"1": [_row("đau ngực", "TRIỆU_CHỨNG", 0)]},
    }

    report = build_phase1_proposal_matrix(
        sources,
        {"1": text},
        source_metadata={"codex": {"model": "codex", "prompt_sha256": "abc"}},
    )
    rows = report["matrix"]
    exact = next(row for row in rows if row["text"] == "đau ngực" and row["type"] == "TRIỆU_CHỨNG")
    diagnosis = next(row for row in rows if row["type"] == "CHẨN_ĐOÁN")
    nested = next(row for row in rows if row["text"] == "ngực")
    source_only = next(row for row in rows if row["text"] == "ho")

    assert exact["source_count"] == 3
    assert exact["all_source_agreement"] is True
    assert exact["status"] == "type_conflict"
    assert diagnosis["status"] == "type_conflict"
    assert nested["status"] == "overlap_agreement"
    assert source_only["status"] == "source_only"
    assert proposal_consensus_keys(report) == {("1", 0, 8, "TRIỆU_CHỨNG")}
    assert report["schema_version"] == "phase1-proposal-matrix.v3"
    assert report["source_metadata"]["codex"]["prompt_sha256"] == "abc"


def test_proposal_matrix_retains_strongest_source_confidence_and_labels() -> None:
    text = "khó thở"
    first = _row("khó thở", "TRIỆU_CHỨNG", 0)
    first.update(
        confidence=0.71,
        source_label="DISEASESYMTOM",
        support_only=True,
    )
    stronger = dict(first)
    stronger["confidence"] = 0.93
    sources = {
        "rule": {"1": [_row("khó thở", "TRIỆU_CHỨNG", 0)]},
        "vietmed": {"1": [first, stronger]},
    }

    report = build_phase1_proposal_matrix(sources, {"1": text})

    row = report["matrix"][0]
    assert row["source_evidence"] == {
        "rule": {
            "confidence": None,
            "source_labels": [],
            "support_only": False,
        },
        "vietmed": {
            "confidence": 0.93,
            "source_labels": ["DISEASESYMTOM"],
            "support_only": True,
        },
    }


def test_proposal_matrix_excludes_invalid_offsets_and_writes_artifacts(tmp_path: Path) -> None:
    sources = {
        "a": {"1": [_row("ho", "TRIỆU_CHỨNG", 0)]},
        "b": {"1": [_row("sai", "TRIỆU_CHỨNG", 0)]},
    }
    report = build_phase1_proposal_matrix(sources, {"1": "ho"})

    assert report["summary"]["invalid_proposal_count"] == 1
    write_phase1_proposal_matrix(report, tmp_path)
    assert (tmp_path / "proposal_matrix.jsonl").exists()
    assert (tmp_path / "review_queue.csv").exists()
    assert (tmp_path / "source_metadata.json").exists()
    blind = json.loads((tmp_path / "codex_blind_queue.jsonl").read_text(encoding="utf-8"))
    assert "proposal" not in blind


@pytest.mark.release
def test_proposal_matrix_cli_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    output = tmp_path / "report"
    for directory in (input_dir, source_a, source_b):
        directory.mkdir()
    (input_dir / "1.txt").write_text("ho", encoding="utf-8")
    payload = json.dumps([_row("ho", "TRIỆU_CHỨNG", 0)], ensure_ascii=False)
    (source_a / "1.json").write_text(payload, encoding="utf-8")
    (source_b / "1.json").write_text(payload, encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_phase1_proposal_matrix.py",
            "--source",
            f"a={source_a}",
            "--source",
            f"b={source_b}",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["all_source_agreement_count"] == 1


def _row(text: str, entity_type: str, start: int) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "candidates": [],
        "position": [start, start + len(text)],
    }
