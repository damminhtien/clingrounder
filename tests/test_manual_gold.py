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
            "100",
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
    assert summary["reviewed_count"] == 45
    assert summary["entity_count"] == 1357
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
        "14.json",
        "15.json",
        "16.json",
        "17.json",
        "18.json",
        "19.json",
        "20.json",
        "21.json",
        "22.json",
        "23.json",
        "24.json",
        "25.json",
        "26.json",
        "27.json",
        "28.json",
        "31.json",
        "32.json",
        "33.json",
        "34.json",
        "35.json",
        "41.json",
        "42.json",
        "43.json",
        "44.json",
        "45.json",
        "94.json",
        "95.json",
        "96.json",
        "97.json",
        "98.json",
        "99.json",
        "100.json",
    ]
