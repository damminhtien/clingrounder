"""Fast contracts for mined-span validation and tokenizer-safe BIO projection."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.training import (
    SpanTrainingEntity,
    SpanTrainingRecord,
    TokenBoundaryAlignmentError,
    build_bio_label_vocabulary,
    compute_bio_span_metrics,
    project_record_to_token_windows,
    scan_span_dataset,
    validate_span_dataset_manifest,
)
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text


def test_span_dataset_manifest_and_train_vocabulary_are_pinned(tmp_path: Path) -> None:
    dataset = tmp_path / "spans.jsonl"
    rows = [
        _raw_record("train-1", "train", "đau ngực", "DISEASE"),
        _raw_record("dev-1", "development", "sốt cao", "DISEASE"),
    ]
    dataset.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "mined-span-dataset.v1",
                "chunk_count": 2,
                "entity_count": 2,
                "output_sha256": sha256_file(dataset),
            }
        ),
        encoding="utf-8",
    )

    summary = scan_span_dataset(dataset)
    validate_span_dataset_manifest(dataset, manifest, summary)

    assert summary.split_record_counts == {"development": 1, "train": 1}
    assert build_bio_label_vocabulary(
        summary,
        train_split="train",
        evaluation_split="development",
    ) == ("O", "B-DISEASE", "I-DISEASE")


def test_train_vocabulary_rejects_label_only_seen_in_evaluation(tmp_path: Path) -> None:
    dataset = tmp_path / "spans.jsonl"
    rows = [
        _raw_record("train-1", "train", "aspirin", "DRUG"),
        _raw_record("dev-1", "development", "đau", "SYMPTOM"),
    ]
    dataset.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = scan_span_dataset(dataset)

    with pytest.raises(ValueError, match="unseen labels.*SYMPTOM"):
        build_bio_label_vocabulary(
            summary,
            train_split="train",
            evaluation_split="development",
        )


def test_projection_owns_overflow_entity_once() -> None:
    record = _record("alpha beta gamma", start=6, end=10)
    tokenizer = _OverflowTokenizer()

    windows = project_record_to_token_windows(
        record,
        tokenizer,
        ("O", "B-DISEASE", "I-DISEASE"),
        max_length=8,
        stride=2,
    )

    assert [window.labels for window in windows] == [
        (-100, 0, 1, -100),
        (-100, -100, 0, -100),
    ]
    assert tokenizer.calls == 1


def test_projection_rejects_token_boundary_drift() -> None:
    record = _record("alpha beta", start=1, end=5)

    with pytest.raises(TokenBoundaryAlignmentError, match="cannot preserve exact boundary"):
        project_record_to_token_windows(
            record,
            _SingleWindowTokenizer(),
            ("O", "B-DISEASE", "I-DISEASE"),
            max_length=8,
            stride=2,
        )


def test_exact_bio_span_metrics_ignore_special_and_overflow_tokens() -> None:
    metrics = compute_bio_span_metrics(
        predicted_label_ids=[[-100, 1, 2, 0, -100], [-100, 1, 0, -100]],
        gold_label_ids=[[-100, 1, 2, 0, -100], [-100, -100, 0, -100]],
        label_vocabulary=("O", "B-DISEASE", "I-DISEASE"),
    )

    assert metrics["span_f1"] == pytest.approx(1.0)
    assert metrics["span_true_positive"] == 1.0
    assert metrics["span_false_positive"] == 0.0


def test_model_training_cli_is_discoverable() -> None:
    args = build_parser().parse_args(
        [
            "model",
            "train-token-classifier",
            "--dataset",
            "spans.jsonl",
            "--dataset-manifest",
            "manifest.json",
            "--model-id",
            "local/model",
            "--revision",
            "deadbeef",
            "--output-dir",
            "outputs/model-run",
        ]
    )

    assert args.handler == "model_train_token_classifier"
    assert args.revision == "deadbeef"


def test_training_import_does_not_import_model_frameworks() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import medical_kg_nlp.training; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules; "
                "assert 'datasets' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr


def _raw_record(record_id: str, split: str, text: str, label: str) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "document_id": f"document-{record_id}",
        "split": split,
        "text": text,
        "text_sha256": sha256_text(text),
        "source_span": [0, len(text)],
        "language": "vi",
        "note_type": "clinical_note",
        "source_artifact_id": f"artifact-{record_id}",
        "entities": [
            {
                "annotation_id": f"annotation-{record_id}",
                "start": 0,
                "end": len(text),
                "source_start": 0,
                "source_end": len(text),
                "text": text,
                "label": label,
            }
        ],
    }


def _record(text: str, *, start: int, end: int) -> SpanTrainingRecord:
    entity = SpanTrainingEntity(
        annotation_id="annotation-1",
        start=start,
        end=end,
        source_start=start,
        source_end=end,
        text=text[start:end],
        label="DISEASE",
    )
    record = SpanTrainingRecord(
        record_id="record-1",
        document_id="document-1",
        split="train",
        text=text,
        text_sha256=sha256_text(text),
        source_span=(0, len(text)),
        language="en",
        note_type="clinical_note",
        source_artifact_id="artifact-1",
        entities=(entity,),
    )
    record.validate()
    return record


class _OverflowTokenizer:
    is_fast = True

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, **kwargs: object) -> dict[str, Any]:
        del text, kwargs
        self.calls += 1
        return {
            "input_ids": [[101, 1, 2, 102], [101, 2, 3, 102]],
            "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
            "offset_mapping": [
                [(0, 0), (0, 5), (6, 10), (0, 0)],
                [(0, 0), (6, 10), (11, 16), (0, 0)],
            ],
        }


class _SingleWindowTokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs: object) -> dict[str, Any]:
        del text, kwargs
        return {
            "input_ids": [101, 1, 2, 102],
            "attention_mask": [1, 1, 1, 1],
            "offset_mapping": [(0, 0), (0, 5), (6, 10), (0, 0)],
        }
