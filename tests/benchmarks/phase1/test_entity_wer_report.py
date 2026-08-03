import json
import subprocess
import sys
from pathlib import Path

import pytest

import yaml

from medical_kg_nlp.benchmarks.phase1.entity_wer_report import (
    build_entity_wer_report,
    write_entity_wer_report,
)


def test_entity_wer_report_tracks_source_boundary_missing_and_spurious(tmp_path: Path) -> None:
    text = "đau ngực dữ dội; CRP âm tính; sốt"
    symptom_start = text.index("đau ngực dữ dội")
    crp_start = text.index("CRP")
    result_start = text.index("âm tính")
    fever_start = text.index("sốt")
    gold = {
        "12": [
            _row("đau ngực dữ dội", "TRIỆU_CHỨNG", symptom_start),
            _row("CRP", "TÊN_XÉT_NGHIỆM", crp_start),
            _row("âm tính", "KẾT_QUẢ_XÉT_NGHIỆM", result_start),
        ]
    }
    type_selected = {
        "12": [
            _row("đau ngực", "TRIỆU_CHỨNG", symptom_start),
            _row("CRP", "TÊN_XÉT_NGHIỆM", crp_start),
        ]
    }
    repeat_recovery = {
        "12": [*type_selected["12"], _row("sốt", "TRIỆU_CHỨNG", fever_start)],
        "99": [_row("unreviewed", "TRIỆU_CHỨNG", 0)],
    }
    policy = {
        "aliases": {
            "strict": {"TRIỆU_CHỨNG": ["đau ngực"], "TÊN_XÉT_NGHIỆM": ["crp"]},
            "context_required": {"KẾT_QUẢ_XÉT_NGHIỆM": ["âm tính"]},
            "reviewed": {},
        },
        "unstable_mentions": [],
        "exclusions": {"strict": {}},
    }

    report = build_entity_wer_report(
        gold_by_doc=gold,
        pred_by_doc=repeat_recovery,
        documents_by_doc={"12": text},
        stages=[("type_selected", type_selected), ("repeat_recovery", repeat_recovery)],
        annotation_policy=policy,
        public_wer=51.6594,
    )
    output_dir = tmp_path / "report"
    write_entity_wer_report(report, output_dir)

    assert report["summary"]["missing_count"] == 1
    assert report["summary"]["spurious_count"] == 1
    assert report["summary"]["predicted_entity_count"] == 3
    assert report["summary"]["boundary_error_count"] == 1
    boundary = report["boundary_errors"][0]
    assert boundary["boundary_kind"] == "end_under"
    assert boundary["missing_suffix"] == "dữ dội"
    assert boundary["source"] == "type_selected"
    sources = {row["source"]: row for row in report["per_source"]}
    assert sources["type_selected"]["matched"] == 2
    assert sources["repeat_recovery"]["spurious"] == 1
    source_ablation = {row["source"]: row for row in report["source_ablation"]}
    assert source_ablation["type_selected"]["source_effect"] == "helpful"
    assert source_ablation["repeat_recovery"]["source_effect"] == "harmful"
    assert report["stage_comparison"][-1]["stage"] == "final"
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "boundary_errors.csv").exists()
    assert (output_dir / "summary.md").exists()


@pytest.mark.release
def test_entity_wer_cli_smoke(tmp_path: Path) -> None:
    text = "ho"
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    document_dir = tmp_path / "documents"
    gold_dir.mkdir()
    pred_dir.mkdir()
    document_dir.mkdir()
    payload = [_row("ho", "TRIỆU_CHỨNG", 0)]
    _write_json(gold_dir / "12.json", payload)
    _write_json(pred_dir / "12.json", payload)
    (document_dir / "12.txt").write_text(text, encoding="utf-8")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "aliases": {
                    "strict": {"TRIỆU_CHỨNG": ["ho"]},
                    "context_required": {},
                    "reviewed": {},
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/phase1/analyze_phase1_entity_wer.py",
            "--gold-dir",
            str(gold_dir),
            "--pred",
            str(pred_dir),
            "--documents",
            str(document_dir),
            "--policy",
            str(policy_path),
            "--stage",
            f"baseline={pred_dir}",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout)["summary"]["micro_wer_proxy"] == 0.0
    assert (output_dir / "per_source.csv").exists()
    assert (output_dir / "per_source_type.csv").exists()
    assert (output_dir / "source_ablation.csv").exists()
    assert (output_dir / "stage_comparison.csv").exists()


def test_entity_wer_boundary_alignment_does_not_pair_distant_mentions() -> None:
    text = "đau" + (" " * 40) + "đau dữ dội"
    gold_start = text.rindex("đau")
    report = build_entity_wer_report(
        gold_by_doc={"1": [_row("đau dữ dội", "TRIỆU_CHỨNG", gold_start)]},
        pred_by_doc={"1": [_row("đau", "TRIỆU_CHỨNG", 0)]},
        documents_by_doc={"1": text},
    )

    assert report["summary"]["missing_count"] == 1
    assert report["summary"]["spurious_count"] == 1
    assert report["summary"]["boundary_error_count"] == 0
    assert report["boundary_errors"] == []


def _row(text: str, entity_type: str, start: int) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "candidates": [],
        "position": [start, start + len(text)],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
