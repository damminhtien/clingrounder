"""Leakage-safe Phase 1 recognition mining integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.benchmarks.phase1.manual_gold import (
    build_manual_gold_split_manifest,
    write_manual_gold_split_manifest,
)
from medical_kg_nlp.benchmarks.phase1.manual_gold_mining import (
    load_phase1_manual_gold_mining_corpus,
)
from medical_kg_nlp.benchmarks.phase1.recognition_mining import (
    Phase1RecognitionMiningConfig,
    run_phase1_recognition_mining,
)
from medical_kg_nlp.mining.io import load_documents


def test_manual_gold_adapter_preserves_offsets_and_split(tmp_path: Path) -> None:
    input_dir, gold_dir, manifest_path, policy_path, baseline_path = _fixture(tmp_path)
    train = load_phase1_manual_gold_mining_corpus(
        input_dir,
        gold_dir,
        manifest_path,
        split="train",
    )
    holdout = load_phase1_manual_gold_mining_corpus(
        input_dir,
        gold_dir,
        manifest_path,
        split="holdout",
    )

    assert len(train.documents) == 2
    assert len(holdout.documents) == 1
    assert all(document.access_class.value == "local_private" for document in train.documents)
    assert all(annotation.metadata["split"] == "train" for annotation in train.annotations)
    for document in (*train.documents, *holdout.documents):
        for annotation in (*train.annotations, *holdout.annotations):
            if annotation.document_id == document.document_id:
                annotation.validate_offsets(document)


def test_recognition_mining_writes_holdout_gate_and_is_idempotent(tmp_path: Path) -> None:
    input_dir, gold_dir, manifest_path, policy_path, baseline_path = _fixture(tmp_path)
    config = Phase1RecognitionMiningConfig(
        input_dir=input_dir,
        gold_dir=gold_dir,
        split_manifest=manifest_path,
        annotation_policy=policy_path,
        baseline_recognition=baseline_path,
        output_root=tmp_path / "runs",
        minimum_exact_f1_gain=0.1,
        minimum_true_positive_gain=1,
    )

    first = run_phase1_recognition_mining(config)
    second = run_phase1_recognition_mining(config)
    run_dir = Path(first["run_dir"])

    assert first == second
    assert first["promotion_gate"]["passed"] is True
    assert first["compilation"]["recognition_concept_count"] == 1
    assert (run_dir / "holdout_benchmark.json").exists()
    concepts = [
        json.loads(line)
        for line in (run_dir / "recognition_concepts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all("document_id" not in concept and "position" not in concept for concept in concepts)
    assert load_documents(run_dir / "train" / "documents.jsonl")[0].access_class.value == (
        "local_private"
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    for document_id in ("1", "2", "11"):
        text = "buồn nôn và tăng huyết áp"
        (input_dir / f"{document_id}.txt").write_text(text, encoding="utf-8")
        start = text.index("buồn nôn")
        gold = [
            {
                "text": "buồn nôn",
                "position": [start, start + len("buồn nôn")],
                "type": "TRIỆU_CHỨNG",
                "assertions": [],
                "candidates": [],
            }
        ]
        (gold_dir / f"{document_id}.json").write_text(
            json.dumps(gold, ensure_ascii=False), encoding="utf-8"
        )
    manifest_path = tmp_path / "holdout_manifest.json"
    write_manual_gold_split_manifest(
        build_manual_gold_split_manifest(gold_dir, input_dir), manifest_path
    )
    policy_path = tmp_path / "annotation_policy.yaml"
    policy_path.write_text(
        "aliases:\n  strict:\n    TRIỆU_CHỨNG:\n      - buồn nôn\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.jsonl"
    baseline_path.write_text(
        json.dumps(
            {
                "concept_id": "ICD10:I10",
                "code": "I10",
                "code_system": "ICD-10",
                "semantic_type": "DISEASE",
                "canonical_name": "tăng huyết áp",
                "source": "fixture",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return input_dir, gold_dir, manifest_path, policy_path, baseline_path
