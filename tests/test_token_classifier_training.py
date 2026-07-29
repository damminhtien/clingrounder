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
    TokenClassifierTrainingConfig,
    TokenBoundaryAlignmentError,
    build_bio_label_vocabulary,
    compute_bio_span_metrics,
    find_unaligned_annotations,
    project_record_to_token_windows,
    scan_span_dataset,
    validate_span_dataset_manifest,
    verify_saved_token_classifier,
)
from medical_kg_nlp.training import huggingface_token_classifier as training_runtime
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match
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


def test_projection_masks_unrepresentable_boundary_without_widening_gold() -> None:
    record = _record("alpha beta", start=1, end=5)
    tokenizer = _SingleWindowTokenizer()

    issues = find_unaligned_annotations(
        record,
        tokenizer,
        max_length=8,
        stride=2,
    )
    windows = project_record_to_token_windows(
        record,
        tokenizer,
        ("O", "B-DISEASE", "I-DISEASE"),
        max_length=8,
        stride=2,
        unaligned_span_policy="mask",
    )

    assert [entity.annotation_id for entity in issues] == ["annotation-1"]
    assert windows[0].labels == (-100, -100, 0, -100)
    assert record.text[record.entities[0].start : record.entities[0].end] == "lpha"


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


def test_cpu_smoke_cli_requires_explicit_cpu_and_verification_text() -> None:
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
            "outputs/model-smoke",
            "--cpu",
            "--cpu-smoke-text",
            "note.txt",
            "--unaligned-span-policy",
            "mask",
        ]
    )

    assert args.cpu is True
    assert args.cpu_smoke_text == "note.txt"
    assert args.unaligned_span_policy == "mask"


def test_saved_model_verification_checks_projected_raw_offsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "final-model"
    model_dir.mkdir()

    class FakeAdapter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def extract(self, source_text: str) -> list[EntityAnnotation]:
            return [
                EntityAnnotation(
                    id="M1",
                    span=(0, 3),
                    text=source_text[:3],
                    normalized_text=normalize_for_match(source_text[:3]),
                    type=EntityType.SYMPTOM,
                    confidence=0.9,
                )
            ]

    monkeypatch.setattr(
        training_runtime,
        "HuggingFaceTokenClassifierAdapter",
        FakeAdapter,
    )

    report = verify_saved_token_classifier(model_dir, "đau ngực")

    assert report["status"] == "passed"
    assert report["projected_entity_count"] == 1
    assert report["offset_mismatch_count"] == 0


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


def test_transformers_five_training_arguments_exclude_removed_overwrite_flag(
    tmp_path: Path,
) -> None:
    config = TokenClassifierTrainingConfig(
        dataset_path=tmp_path / "spans.jsonl",
        dataset_manifest_path=tmp_path / "manifest.json",
        output_dir=tmp_path / "model",
        model_id="local/model",
        revision="deadbeef",
        overwrite_output=True,
    )

    arguments = training_runtime._training_argument_kwargs(  # noqa: SLF001
        config,
        has_evaluation=True,
    )

    assert "overwrite_output_dir" not in arguments
    assert arguments["eval_strategy"] == "epoch"
    assert arguments["load_best_model_at_end"] is True


def test_local_initialization_model_is_content_verified(tmp_path: Path) -> None:
    model_dir = tmp_path / "dapt-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    fingerprint = training_runtime.fingerprint_model_directory(model_dir)
    config = TokenClassifierTrainingConfig(
        dataset_path=tmp_path / "spans.jsonl",
        dataset_manifest_path=tmp_path / "manifest.json",
        output_dir=tmp_path / "model",
        model_id="upstream/model",
        revision="a" * 40,
        initialization_model_path=model_dir,
        initialization_model_fingerprint=fingerprint,
    )

    assert training_runtime._verify_initialization_model(config) == str(model_dir)

    (model_dir / "config.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        training_runtime._verify_initialization_model(config)


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
