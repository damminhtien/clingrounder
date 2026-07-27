"""Leakage and reproducibility gates for Qwen instruction datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.qwen_dataset import (
    Phase1QwenDatasetConfig,
    build_phase1_qwen_instruction_dataset,
)
from medical_kg_nlp.mining.io import write_json, write_jsonl


def test_qwen_dataset_builds_train_and_development_without_offsets_in_target(
    tmp_path: Path,
) -> None:
    spans = tmp_path / "spans.jsonl"
    rows = [
        _span_row("1", "train", "Bệnh nhân ho", 10, 12, "SYMPTOM"),
        _span_row("2", "development", "Có tăng huyết áp", 3, 16, "DISEASE"),
    ]
    spans_sha256 = write_jsonl(spans, rows)
    manifest = tmp_path / "source-manifest.json"
    _write_source_manifest(manifest, spans_sha256)

    report = build_phase1_qwen_instruction_dataset(
        Phase1QwenDatasetConfig(
            spans_path=spans,
            spans_manifest_path=manifest,
            output_dir=tmp_path / "output",
        )
    )

    assert report["outputs"]["extraction"]["split_counts"] == {
        "development": 1,
        "train": 1,
    }
    output_rows = [
        json.loads(line)
        for line in (tmp_path / "output/extraction.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assistant = json.loads(output_rows[0]["messages"][-1]["content"])
    assert assistant["entities"] == [
        {
            "confidence": 1.0,
            "left_context": "",
            "right_context": "",
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
        }
    ]
    assert "position" not in assistant["entities"][0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_id", "phase1_round2:1"),
        ("source_artifact_id", "quarantine:leaked"),
        ("record_id", "input_part2:1"),
    ],
)
def test_qwen_dataset_rejects_round2_and_quarantined_sources(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    spans = tmp_path / "spans.jsonl"
    train = _span_row("1", "train", "Bệnh nhân ho", 10, 12, "SYMPTOM")
    train[field] = value
    rows = [
        train,
        _span_row("2", "development", "Có sốt", 3, 6, "SYMPTOM"),
    ]
    spans_sha256 = write_jsonl(spans, rows)
    manifest = tmp_path / "source-manifest.json"
    _write_source_manifest(manifest, spans_sha256)

    with pytest.raises(ValueError, match="Forbidden Qwen supervision source"):
        build_phase1_qwen_instruction_dataset(
            Phase1QwenDatasetConfig(
                spans_path=spans,
                spans_manifest_path=manifest,
                output_dir=tmp_path / "output",
            )
        )


def test_hard_negative_builder_rejects_development_predictions(tmp_path: Path) -> None:
    spans = tmp_path / "spans.jsonl"
    rows = [
        _span_row("1", "train", "Bệnh nhân ho", 10, 12, "SYMPTOM"),
        _span_row("2", "development", "Có sốt", 3, 6, "SYMPTOM"),
    ]
    spans_sha256 = write_jsonl(spans, rows)
    manifest = tmp_path / "source-manifest.json"
    _write_source_manifest(manifest, spans_sha256)
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        (
            {
                "document_id": "2",
                "entities": [
                    {
                        "span": [0, 2],
                        "text": "Có",
                        "type": "SYMPTOM",
                        "confidence": 0.9,
                    }
                ],
            },
        ),
    )

    with pytest.raises(ValueError, match="train documents only"):
        build_phase1_qwen_instruction_dataset(
            Phase1QwenDatasetConfig(
                spans_path=spans,
                spans_manifest_path=manifest,
                output_dir=tmp_path / "output",
                hard_negative_predictions_path=predictions,
            )
        )


def _span_row(
    document_id: str,
    split: str,
    text: str,
    start: int,
    end: int,
    label: str,
) -> dict[str, object]:
    import hashlib

    return {
        "record_id": f"span-record:{document_id}",
        "document_id": f"phase1-manual-gold:{document_id}",
        "source_artifact_id": "phase1-manual-gold:" + "a" * 64,
        "split": split,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "entities": [
            {
                "start": start,
                "end": end,
                "text": text[start:end],
                "label": label,
            }
        ],
    }


def _write_source_manifest(path: Path, spans_sha256: str) -> None:
    write_json(
        path,
        {
            "schema_version": "mined-span-dataset.v1",
            "output_sha256": spans_sha256,
            "augmentation": {
                "round2_included": False,
                "quarantined_data_included": False,
            },
        },
    )
