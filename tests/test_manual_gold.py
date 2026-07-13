import json
import subprocess
import sys


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
