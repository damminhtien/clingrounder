import json
import subprocess
import sys


def test_validate_manual_gold_allows_incomplete_review_batch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_manual_gold.py",
            "--allow-incomplete",
            "--expected-count",
            "13",
            "--input-dir",
            "data/raw/input",
            "--gold-dir",
            "data/manual_gold",
            "--dictionary",
            "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)

    assert summary["valid"] is True
    assert summary["reviewed_count"] == 13
    assert summary["entity_count"] == 402
    assert summary["reviewed_files"] == [
        "1.json",
        "2.json",
        "3.json",
        "4.json",
        "5.json",
        "6.json",
        "7.json",
        "8.json",
        "9.json",
        "10.json",
        "11.json",
        "12.json",
        "13.json",
    ]
