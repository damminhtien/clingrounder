import json
import subprocess
import sys
from pathlib import Path

import pytest

from medical_kg_nlp.evaluation.manual_gold import (
    build_manual_gold_split_manifest,
    verify_manual_gold_split_manifest,
)


@pytest.mark.private
def test_validate_complete_manual_gold_batch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_manual_gold.py",
            "--expected-count",
            "100",
            "--input-dir",
            "data/raw/input",
            "--gold-dir",
            "data/manual_gold",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)

    assert summary["valid"] is True
    assert summary["reviewed_count"] == 100
    assert summary["missing_count"] == 0
    assert summary["entity_count"] == 2777
    assert summary["reviewed_files"] == [f"{index}.json" for index in range(1, 101)]


def test_manual_gold_split_manifest_detects_label_or_source_drift(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    document_dir = tmp_path / "input"
    gold_dir.mkdir()
    document_dir.mkdir()
    (gold_dir / "1.json").write_text("[]\n", encoding="utf-8")
    (document_dir / "1.txt").write_text("Đau đầu", encoding="utf-8")

    manifest = build_manual_gold_split_manifest(gold_dir, document_dir)
    verify_manual_gold_split_manifest(manifest, gold_dir, document_dir)
    assert manifest["corpus"]["document_count"] == 1
    assert manifest["splits"]["train"]["document_ids"] == ["1"]

    (gold_dir / "1.json").write_text(
        json.dumps([{"text": "Đau đầu"}], ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no longer matches"):
        verify_manual_gold_split_manifest(manifest, gold_dir, document_dir)
