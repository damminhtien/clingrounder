from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("source_count", [2, 3])
@pytest.mark.release
def test_top10_probe_suite_cli_builds_isolated_validated_artifacts(
    tmp_path: Path,
    source_count: int,
) -> None:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    base_dir = tmp_path / "base"
    qwen_dir = tmp_path / "qwen"
    pipeline_dir = tmp_path / "pipeline"
    codex_dir = tmp_path / "codex"
    for directory in (input_dir, gold_dir, base_dir, qwen_dir, pipeline_dir, codex_dir):
        directory.mkdir()

    texts = {
        "1": "Tiền sử bệnh:\nTăng huyết áp. Kali: 2.4. Dùng 80mg.",
        "2": "Tăng huyết áp",
    }
    predictions: dict[str, list[dict[str, object]]] = {}
    gold: dict[str, list[dict[str, object]]] = {}
    for document_id, text in texts.items():
        diagnosis_start = text.index("Tăng huyết áp")
        prediction_rows = [_row("Tăng huyết áp", "CHẨN_ĐOÁN", diagnosis_start)]
        gold_rows = [
            _row(
                "Tăng huyết áp",
                "CHẨN_ĐOÁN",
                diagnosis_start,
                assertions=["isHistorical"] if document_id == "1" else [],
                candidates=["I10"],
            )
        ]
        if document_id == "1":
            prediction_rows.extend(
                [
                    _row("Kali", "TÊN_XÉT_NGHIỆM", text.index("Kali")),
                    _row("2.4", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("2.4")),
                    _row("80mg", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("80mg")),
                ]
            )
            gold_rows.extend(prediction_rows[1:3])
        predictions[document_id] = prediction_rows
        gold[document_id] = gold_rows
        (input_dir / f"{document_id}.txt").write_text(text, encoding="utf-8")
        for directory in (base_dir, qwen_dir, pipeline_dir, codex_dir):
            _write_json(directory / f"{document_id}.json", prediction_rows)
        _write_json(gold_dir / f"{document_id}.json", gold_rows)

    manifest = tmp_path / "review_manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "document_id": document_id,
                    "gold_file": str(gold_dir / f"{document_id}.json"),
                    "source_file": str(input_dir / f"{document_id}.txt"),
                    "entity_count": len(gold[document_id]),
                    "review_candidates": [],
                    "guideline_notes": [],
                },
                ensure_ascii=False,
            )
            + "\n"
            for document_id in texts
        ),
        encoding="utf-8",
    )
    dictionary = tmp_path / "dictionary.jsonl"
    dictionary.write_text(
        json.dumps(
            {
                "concept_id": "ICD10:I10",
                "code": "I10",
                "code_system": "ICD-10",
                "canonical_name": "Tăng huyết áp",
                "semantic_type": "DISEASE",
                "aliases": [],
                "synonyms": [],
                "abbreviations": [],
                "source": "icd10_vn_tt06_2026",
                "source_ids": ["icd10_vn_tt06_2026"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "runs"

    command = [
        sys.executable,
        "scripts/run_phase1_top10_probes.py",
        "--base",
        str(base_dir),
        "--source",
        f"qwen={qwen_dir}",
        "--source",
        f"pipeline={pipeline_dir}",
    ]
    if source_count == 3:
        command.extend(["--source", f"codex={codex_dir}", "--full-diagnostic"])
    command.extend(
        [
            "--input-dir",
            str(input_dir),
            "--gold-dir",
            str(gold_dir),
            "--review-manifest",
            str(manifest),
            "--dictionary",
            str(dictionary),
            "--output-root",
            str(output_root),
            "--journal-dir",
            str(tmp_path / "journal"),
        ]
    )
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    run_dir = Path(payload["run_dir"])
    manifest_payload = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    variants = {row["name"]: row for row in manifest_payload["variants"]}
    assert manifest_payload["holdout_status"] == "sealed"
    assert manifest_payload["tri_source_ready"] is (source_count == 3)
    expected_candidate_ready = source_count == 3
    assert manifest_payload["candidate_consensus_ready"] is expected_candidate_ready
    assert (manifest_payload["candidate_consensus_key_count"] > 0) is expected_candidate_ready
    assert manifest_payload["minimum_candidate_proposal_sources"] == 3
    assert (manifest_payload["candidate_probe_blocked_reason"] is None) is expected_candidate_ready
    assert variants["E_LAB"]["changed"]["entity_removed"] == 1
    assert variants["E_LAB"]["local_safety_gate_passed"] is True
    assert variants["E_LAB"]["probe_ready"] is True
    assert variants["A_HIST"]["changed"]["assertion_changed"] == 1
    assert "A_FAM_EXT" in variants
    assert variants["C_ICD_ONE"]["changed"].get("candidate_changed", 0) == (
        2 if expected_candidate_ready else 0
    )
    assert variants["C_ICD_ONE"]["probe_ready"] is expected_candidate_ready
    assert (run_dir / "variants" / "E_LAB" / "output.zip").exists()
    assert (run_dir / "boundary_rule_candidates.yaml").exists()
    assert (run_dir / "proposals" / "review_queue.csv").exists()
    assert (tmp_path / "journal" / "phase1_top10_probe_runs.jsonl").exists()
    if source_count == 3:
        assert manifest_payload["best_full_diagnostic"]["name"] in {
            "FULL_ICD_DIAGNOSTIC",
            "FULL_RX_DIAGNOSTIC",
            "FULL_ALL_CANDIDATES_DIAGNOSTIC",
            "FULL_ALL_MODULES_DIAGNOSTIC",
        }
        assert manifest_payload["best_full_diagnostic"]["submission_recommended"] is False


def _row(
    text: str,
    entity_type: str,
    start: int,
    *,
    assertions: list[str] | None = None,
    candidates: list[str] | None = None,
) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "assertions": assertions or [],
        "candidates": candidates or [],
        "position": [start, start + len(text)],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
