"""Contracts for the pinned learned joint-span submission runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.joint_span_run import load_phase1_joint_span_run_spec
from medical_kg_nlp.utils.hashing import sha256_directory, sha256_file


def test_joint_span_run_spec_requires_generated_model_source_and_calibration(tmp_path: Path) -> None:
    root, paths = _write_fixture(root=tmp_path / "run-root")
    config = root / "joint-span.yaml"
    config.write_text(json.dumps(_config_payload(paths), ensure_ascii=False), encoding="utf-8")

    spec = load_phase1_joint_span_run_spec(config)

    assert spec.model_sources[0].role.value == "llm"
    assert spec.candidate_source_priority == ("rule", "medication_parser", "qwen")
    assert spec.calibration.sha256 == sha256_file(paths["calibration"])


def test_joint_span_run_spec_rejects_unpinned_calibration(tmp_path: Path) -> None:
    root, paths = _write_fixture(root=tmp_path / "run-root")
    payload = _config_payload(paths)
    payload.pop("calibration")
    config = root / "joint-span.yaml"
    config.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration must be a mapping"):
        load_phase1_joint_span_run_spec(config)


def _write_fixture(*, root: Path) -> tuple[Path, dict[str, Path]]:
    root.mkdir()
    documents = root / "round2.jsonl"
    documents.write_text("{}\n", encoding="utf-8")
    budget = root / "budget.yaml"
    budget.write_text("schema_version: inference-model-budget-spec.v1\n", encoding="utf-8")
    dictionary = root / "dictionary.jsonl"
    dictionary.write_text("\n", encoding="utf-8")
    verifier = root / "verifier"
    verifier.mkdir()
    (verifier / "config.json").write_text("{}", encoding="utf-8")
    qwen = root / "qwen"
    (qwen / "consensus").mkdir(parents=True)
    (qwen / "consensus" / "1.json").write_text("[]\n", encoding="utf-8")
    calibration = root / "calibration.json"
    training_manifest = root / "verifier-training-manifest.json"
    family = "d" * 64
    training_manifest.write_text(
        json.dumps(
            {
                "schema_version": "phase1-joint-span-verifier-training.v2",
                "model": {"fingerprint": sha256_directory(verifier)},
                "training_family_fingerprint": family,
            }
        ),
        encoding="utf-8",
    )
    calibration.write_text(
        json.dumps(_calibration_payload(family), ensure_ascii=False),
        encoding="utf-8",
    )
    return root, {
        "documents": documents,
        "budget": budget,
        "dictionary": dictionary,
        "verifier": verifier,
        "qwen": qwen,
        "calibration": calibration,
        "training_manifest": training_manifest,
    }


def _calibration_payload(training_family_fingerprint: str) -> dict[str, object]:
    return {
        "schema_version": "phase1-joint-span-calibration.v2",
        "training_family_fingerprint": training_family_fingerprint,
        "oof_observations_sha256": "b" * 64,
        "fold_assignment_sha256": "c" * 64,
        "false_positive_cost": 1.0,
        "points": [
            {
                "genre": genre,
                "type": entity_type,
                "slope": 1.0,
                "intercept": 0.0,
                "threshold": 0.5,
                "positive_count": 1,
                "negative_count": 1,
            }
            for genre in ("clinical", "educational", "qa")
            for entity_type in (
                "CHẨN_ĐOÁN",
                "THUỐC",
                "TRIỆU_CHỨNG",
                "TÊN_XÉT_NGHIỆM",
                "KẾT_QUẢ_XÉT_NGHIỆM",
            )
        ],
    }


def _config_payload(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "schema_version": "phase1-joint-span-run-spec.v4",
        "run_root": ".",
        "documents": {
            "path": paths["documents"].name,
            "sha256": sha256_file(paths["documents"]),
            "source_archive_sha256": "a" * 64,
            "expected_count": 100,
        },
        "budget_spec": paths["budget"].name,
        "verifier": {
            "path": paths["verifier"].name,
            "sha256": sha256_directory(paths["verifier"]),
            "model_id": "FacebookAI/xlm-roberta-base",
            "training_manifest": {
                "path": paths["training_manifest"].name,
                "sha256": sha256_file(paths["training_manifest"]),
            },
        },
        "calibration": {
            "path": paths["calibration"].name,
            "sha256": sha256_file(paths["calibration"]),
        },
        "model_sources": [
            {
                "name": "qwen",
                "role": "llm",
                "path": paths["qwen"].name,
                "sha256": sha256_directory(paths["qwen"]),
            }
        ],
        "dictionaries": [
            {"path": paths["dictionary"].name, "sha256": sha256_file(paths["dictionary"])}
        ],
        "candidate_source_priority": ["rule", "medication_parser", "qwen"],
        "assertion_regimes": ["negation", "history"],
        "candidate_policy": "rx_unique_keep_icd",
        "output_root": "outputs",
        "run_label": "joint-span-test",
    }
