"""Pinned run-spec and CLI contracts for max-score composition."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.max_score_run import (
    PinnedPhase1Artifact,
    load_phase1_max_score_run_spec,
)
from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole


def test_max_score_run_spec_resolves_every_path_below_run_root(
    tmp_path: Path,
) -> None:
    config = _write_spec(tmp_path)

    spec = load_phase1_max_score_run_spec(config)

    assert spec.run_root == tmp_path
    assert spec.documents.path == tmp_path / "documents.jsonl"
    assert spec.budget_spec_path == tmp_path / "budget.yaml"
    assert tuple(source.role for source in spec.sources) == (
        ProposalSourceRole.RULE,
        ProposalSourceRole.LLM,
    )
    assert spec.candidate_source_priority == ("qwen", "rule")
    assert spec.assertion_regimes == ("negation", "history")
    assert spec.boundary_verifier is None
    assert spec.boundary_policy.mode == "disabled"
    assert spec.proposal_thresholds == (
        ("CHẨN_ĐOÁN", 0.24),
        ("KẾT_QUẢ_XÉT_NGHIỆM", 0.16),
        ("THUỐC", 0.03),
        ("TRIỆU_CHỨNG", 0.02),
        ("TÊN_XÉT_NGHIỆM", 0.09),
    )


def test_max_score_run_spec_requires_artifact_for_conservative_boundary_policy(
    tmp_path: Path,
) -> None:
    config = _write_spec(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8")
        .replace(
            "boundary_verifier: null",
            f"boundary_verifier:\n  path: boundary-verifier.json\n  sha256: {'d' * 64}",
        )
        .replace("mode: disabled", "mode: conservative_replacement"),
        encoding="utf-8",
    )

    spec = load_phase1_max_score_run_spec(config)

    assert spec.boundary_verifier is not None
    assert spec.boundary_verifier.path == tmp_path / "boundary-verifier.json"
    assert spec.boundary_policy.mode == "conservative_replacement"


def test_pinned_artifact_rejects_replaced_bytes(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{}\n", encoding="utf-8")
    artifact = PinnedPhase1Artifact(
        path=artifact_path,
        sha256=hashlib.sha256(b"original").hexdigest(),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        artifact.verify(name="fixture")


def test_max_score_cli_uses_one_pinned_config() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "round2",
            "max-score",
            "--config",
            "configs/max-score.yaml",
        ]
    )

    assert args.handler == "benchmark_phase1_round2_max_score"
    assert args.config == "configs/max-score.yaml"


def _write_spec(root: Path) -> Path:
    sha = "a" * 64
    config = root / "max-score.yaml"
    config.write_text(
        f"""
schema_version: phase1-max-score-run-spec.v2
run_root: .
documents:
  path: documents.jsonl
  sha256: {sha}
  source_archive_sha256: {"b" * 64}
  expected_count: 100
budget_spec: budget.yaml
verifier:
  path: verifier.json
  sha256: {"c" * 64}
boundary_verifier: null
boundary_policy:
  mode: disabled
  require_same_type: true
  require_base_selected: true
  preserve_unmodified_identity: true
proposal_thresholds:
  CHẨN_ĐOÁN: 0.24
  KẾT_QUẢ_XÉT_NGHIỆM: 0.16
  THUỐC: 0.03
  TRIỆU_CHỨNG: 0.02
  TÊN_XÉT_NGHIỆM: 0.09
sources:
  - name: rule
    role: rule
    path: rule.zip
    sha256: {"d" * 64}
  - name: qwen
    role: llm
    path: qwen.zip
    sha256: {"e" * 64}
dictionaries:
  - path: terminology.jsonl
    sha256: {"f" * 64}
candidate_source_priority: [qwen, rule]
assertion_regimes: [negation, history]
candidate_policy: rx_unique_keep_icd
output_root: outputs
run_label: max-score
""".lstrip(),
        encoding="utf-8",
    )
    return config
