from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.evaluation.phase1_probe_gate import (
    append_public_probe_journal,
    evaluate_public_probe_promotion,
)


BASELINE = {
    "primaryScore": 38.7975,
    "metrics": {"WER": 51.6594, "J_assertion": 40.9174, "J_candidates": 30.0503},
}


def test_candidate_probe_requires_target_and_total_gain_with_isolation() -> None:
    trial = {
        "primaryScore": 39.1,
        "metrics": {"WER": 51.6594, "J_assertion": 40.9174, "J_candidates": 30.7},
    }
    gate = evaluate_public_probe_promotion(BASELINE, trial, module="candidate")
    assert gate["passed"] is True

    regressed = {
        "primaryScore": 39.1,
        "metrics": {"WER": 51.7, "J_assertion": 40.9174, "J_candidates": 30.7},
    }
    assert evaluate_public_probe_promotion(BASELINE, regressed, module="candidate")["passed"] is False


def test_entity_and_assertion_probe_target_gates() -> None:
    entity = {
        "primaryScore": 39.0,
        "metrics": {"WER": 50.9, "J_assertion": 41.0, "J_candidates": 30.2},
    }
    assertion = {
        "primaryScore": 39.0,
        "metrics": {"WER": 51.6594, "J_assertion": 41.5, "J_candidates": 30.0503},
    }
    assert evaluate_public_probe_promotion(BASELINE, entity, module="entity")["passed"] is True
    assert evaluate_public_probe_promotion(BASELINE, assertion, module="assertion")["passed"] is True

    entity_regression = {
        "primaryScore": 39.0,
        "metrics": {"WER": 50.9, "J_assertion": 40.0, "J_candidates": 30.2},
    }
    assert (
        evaluate_public_probe_promotion(BASELINE, entity_regression, module="entity")["passed"]
        is False
    )


def test_probe_journal_appends_sha_and_markdown(tmp_path: Path) -> None:
    artifact = tmp_path / "output.zip"
    artifact.write_bytes(b"zip")
    gate = evaluate_public_probe_promotion(
        BASELINE,
        {
            "primaryScore": 39.0,
            "metrics": {"WER": 50.9, "J_assertion": 40.9174, "J_candidates": 30.0503},
        },
        module="entity",
    )
    record = append_public_probe_journal(
        gate,
        tmp_path / "journal",
        probe_name="E-LAB",
        artifact_path=artifact,
        policy_diff={"lab_gate": True},
    )

    assert record["decision"] == "keep"
    rows = (tmp_path / "journal" / "phase1_top10_public_probes.jsonl").read_text(
        encoding="utf-8"
    )
    assert json.loads(rows)["artifact_sha256"]
    assert "E-LAB" in (tmp_path / "journal" / "phase1_top10_public_probes.md").read_text(
        encoding="utf-8"
    )
