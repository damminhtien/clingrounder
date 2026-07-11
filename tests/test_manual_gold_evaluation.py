import json
import subprocess
import sys
from pathlib import Path

from medical_kg_nlp.evaluation.manual_gold import (
    compare_manual_gold_gate,
    evaluate_manual_gold,
    manual_gold_split,
    write_manual_gold_report,
)


def test_manual_gold_split_is_stable() -> None:
    assert manual_gold_split("11") == "holdout"
    assert manual_gold_split("12") == "train"


def test_manual_gold_report_tracks_null_and_positive_metrics(tmp_path: Path) -> None:
    gold = {
        "11": [_row("ho", "TRIỆU_CHỨNG", assertions=["isNegated"])],
        "12": [_row("sốt", "TRIỆU_CHỨNG")],
    }
    pred = {
        "11": [_row("ho", "TRIỆU_CHỨNG", assertions=[])],
        "12": [_row("sốt", "TRIỆU_CHỨNG")],
    }

    report = evaluate_manual_gold(gold, pred)
    assertion_metrics = report["splits"]["all"]["selective_prediction"]["assertions"]
    output_dir = tmp_path / "report"
    write_manual_gold_report(report, output_dir)

    assert report["splits"]["holdout"]["document_ids"] == ["11"]
    assert assertion_metrics["prediction_coverage"] == 0.0
    assert assertion_metrics["positive_recall"] == 0.0
    assert assertion_metrics["null_accuracy"] == 1.0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "errors.csv").exists()
    assert (output_dir / "errors.jsonl").exists()
    assert (output_dir / "summary.md").exists()


def test_manual_gold_gate_applies_holdout_thresholds() -> None:
    report = {
        "splits": {
            "holdout": {
                "metrics": {"score": 37.0, "text_score": 0.52},
                "error_counts": {
                    "phase1_missing_entity": 160,
                    "phase1_spurious_entity": 25,
                    "phase1_text_boundary": 70,
                },
            }
        }
    }
    baseline = json.loads(Path("data/manual_gold/entity_only_baseline.json").read_text(encoding="utf-8"))

    gate = compare_manual_gold_gate(report, baseline)

    assert gate["passed"] is True


def test_manual_gold_cli_smoke(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()
    payload = [_row("ho", "TRIỆU_CHỨNG")]
    (gold_dir / "11.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (gold_dir / "entity_only_baseline.json").write_text('{"metadata": true}', encoding="utf-8")
    (pred_dir / "11.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_phase1_manual_gold.py",
            "--gold-dir",
            str(gold_dir),
            "--pred-dir",
            str(pred_dir),
            "--output-dir",
            str(output_dir),
            "--baseline",
            str(tmp_path / "missing.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout)["holdout"]["score"] == 100.0
    assert (output_dir / "metrics.json").exists()


def _row(
    text: str,
    entity_type: str,
    *,
    assertions: list[str] | None = None,
    candidates: list[str] | None = None,
) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "assertions": assertions or [],
        "candidates": candidates or [],
        "position": [0, len(text)],
    }
