from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from clingrounder.evaluation.data_profile import profile_paths


def test_profile_paths_reports_core_distributions() -> None:
    profile = profile_paths(
        "data/samples/sample_notes.jsonl",
        "data/samples/gold.jsonl",
        dictionary_path="data/dictionaries/seed_concepts.jsonl",
        reference_gold_path="data/samples/gold.jsonl",
        top_k=5,
    )

    assert profile["documents"]["count"] == 1
    assert profile["entities"]["count"] == 10
    assert profile["relations"]["count"] == 6
    assert profile["dictionary_coverage"]["coverage"] == 1.0
    assert profile["offsets"]["issue_count"] == 0
    assert profile["code_overlap"]["unseen_code_count"] == 0
    entity_types = {item["key"]: item["count"] for item in profile["entities"]["by_type"]}
    assert entity_types["DISEASE"] == 4
    context_cues = {item["key"]: item["count"] for item in profile["context_cues"]}
    assert context_cues["negation:không ghi nhận"] == 1


@pytest.mark.integration
def test_profile_data_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    json_path = tmp_path / "profile.json"
    markdown_path = tmp_path / "profile.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/profile_data.py",
            "--documents",
            "data/samples/sample_notes.jsonl",
            "--gold",
            "data/samples/gold.jsonl",
            "--dictionary",
            "data/dictionaries/seed_concepts.jsonl",
            "--output",
            str(json_path),
            "--markdown",
            str(markdown_path),
            "--top-k",
            "5",
        ],
        check=True,
    )

    profile = json.loads(json_path.read_text(encoding="utf-8"))
    assert profile["entities"]["coded_count"] == 9
    assert "# Data Profile" in markdown_path.read_text(encoding="utf-8")
