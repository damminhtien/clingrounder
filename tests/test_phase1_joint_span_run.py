"""Contracts for the pinned learned joint-span submission runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.joint_span_run import (
    load_phase1_joint_span_run_spec,
)
from medical_kg_nlp.utils.hashing import sha256_directory, sha256_file


def test_joint_span_run_spec_requires_generated_and_model_source_priority(tmp_path: Path) -> None:
    root = tmp_path / "run-root"
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

    config = root / "joint-span.yaml"
    config.write_text(
        json.dumps(
            {
                "schema_version": "phase1-joint-span-run-spec.v2",
                "run_root": ".",
                "documents": {
                    "path": "round2.jsonl",
                    "sha256": sha256_file(documents),
                    "source_archive_sha256": "a" * 64,
                    "expected_count": 100,
                },
                "budget_spec": "budget.yaml",
                "verifier": {
                    "path": "verifier",
                    "sha256": sha256_directory(verifier),
                    "model_id": "FacebookAI/xlm-roberta-base",
                },
                "selection_policy": {
                    "genre_type_thresholds": _genre_type_thresholds(),
                    "false_positive_cost": 1.0,
                },
                "model_sources": [
                    {
                        "name": "qwen",
                        "role": "llm",
                        "path": "qwen",
                        "sha256": sha256_directory(qwen),
                    }
                ],
                "dictionaries": [
                    {"path": "dictionary.jsonl", "sha256": sha256_file(dictionary)}
                ],
                "candidate_source_priority": ["rule", "medication_parser", "qwen"],
                "assertion_regimes": ["negation", "history"],
                "candidate_policy": "rx_unique_keep_icd",
                "output_root": "outputs",
                "run_label": "joint-span-test",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    spec = load_phase1_joint_span_run_spec(config)

    assert spec.model_sources[0].role.value == "llm"
    assert spec.candidate_source_priority == ("rule", "medication_parser", "qwen")
    assert spec.selection_policy.false_positive_cost == 1.0


def test_joint_span_run_spec_rejects_missing_submission_genre_threshold(tmp_path: Path) -> None:
    root = tmp_path / "run-root"
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
    config = root / "joint-span.yaml"
    config.write_text(
        json.dumps(
            _config_payload(documents, budget, dictionary, verifier, qwen),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lacks calibrated thresholds"):
        load_phase1_joint_span_run_spec(config)


def _genre_type_thresholds() -> list[dict[str, object]]:
    return [
        {"genre": genre, "type": entity_type, "threshold": 0.5}
        for genre in ("clinical", "educational", "qa")
        for entity_type in (
            "CHẨN_ĐOÁN",
            "THUỐC",
            "TRIỆU_CHỨNG",
            "TÊN_XÉT_NGHIỆM",
            "KẾT_QUẢ_XÉT_NGHIỆM",
        )
    ]


def _config_payload(
    documents: Path,
    budget: Path,
    dictionary: Path,
    verifier: Path,
    qwen: Path,
) -> dict[str, object]:
    thresholds = _genre_type_thresholds()
    thresholds.pop()
    return {
        "schema_version": "phase1-joint-span-run-spec.v2",
        "run_root": ".",
        "documents": {
            "path": documents.name,
            "sha256": sha256_file(documents),
            "source_archive_sha256": "a" * 64,
            "expected_count": 100,
        },
        "budget_spec": budget.name,
        "verifier": {
            "path": verifier.name,
            "sha256": sha256_directory(verifier),
            "model_id": "FacebookAI/xlm-roberta-base",
        },
        "selection_policy": {
            "genre_type_thresholds": thresholds,
            "false_positive_cost": 1.0,
        },
        "model_sources": [
            {
                "name": "qwen",
                "role": "llm",
                "path": qwen.name,
                "sha256": sha256_directory(qwen),
            }
        ],
        "dictionaries": [{"path": dictionary.name, "sha256": sha256_file(dictionary)}],
        "candidate_source_priority": ["rule", "medication_parser", "qwen"],
        "assertion_regimes": ["negation", "history"],
        "candidate_policy": "rx_unique_keep_icd",
        "output_root": "outputs",
        "run_label": "joint-span-test",
    }
