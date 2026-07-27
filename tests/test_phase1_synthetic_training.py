"""Contracts for bounded, train-only user synthetic supervision."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from medical_kg_nlp.benchmarks.phase1.synthetic_training import (
    Phase1SyntheticTrainingConfig,
    build_phase1_synthetic_training_dataset,
)
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.training.span_dataset import iter_span_training_records
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text


def test_synthetic_builder_exports_train_only_and_caps_fraction(
    tmp_path: Path,
) -> None:
    human_spans, human_manifest = _human_dataset(tmp_path, train_count=3)
    archive = _synthetic_archive(tmp_path, train_count=5)

    report = build_phase1_synthetic_training_dataset(
        Phase1SyntheticTrainingConfig(
            archive_path=archive,
            expected_archive_sha256=sha256_file(archive),
            human_spans_path=human_spans,
            human_manifest_path=human_manifest,
            output_dir=tmp_path / "output",
            max_synthetic_train_fraction=0.4,
        )
    )

    # floor(0.4 * 3 / 0.6) = 2 synthetic records.
    assert report["dataset"]["synthetic_train_record_count"] == 2
    assert report["dataset"]["synthetic_train_fraction"] == pytest.approx(0.4)
    records = list(iter_span_training_records(tmp_path / "output/spans.jsonl"))
    synthetic = [
        record
        for record in records
        if record.source_artifact_id.startswith("user-synthetic:")
    ]
    assert len(synthetic) == 2
    assert all(record.split == "train" for record in synthetic)
    assert all(record.note_type in {"qa_advice", "structured_discharge"} for record in synthetic)
    assert all(entity.label == "SYMPTOM" for record in synthetic for entity in record.entities)

    manifest = json.loads(
        (tmp_path / "output/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["augmentation"]["synthetic_development_record_count"] == 0
    assert manifest["augmentation"]["candidate_labels_exported"] is False
    assert manifest["augmentation"]["assertion_labels_exported"] is False


def test_synthetic_builder_validates_unexported_source_splits(
    tmp_path: Path,
) -> None:
    human_spans, human_manifest = _human_dataset(tmp_path, train_count=3)
    archive = _synthetic_archive(tmp_path, train_count=2, corrupt_test_offset=True)

    with pytest.raises(ValueError, match="violates raw offsets"):
        build_phase1_synthetic_training_dataset(
            Phase1SyntheticTrainingConfig(
                archive_path=archive,
                expected_archive_sha256=sha256_file(archive),
                human_spans_path=human_spans,
                human_manifest_path=human_manifest,
                output_dir=tmp_path / "output",
            )
        )


def _human_dataset(
    root: Path,
    *,
    train_count: int,
) -> tuple[Path, Path]:
    spans = root / "human-spans.jsonl"
    rows = [
        _span_row(f"train-{index}", "train", f"Bệnh nhân ho {index}")
        for index in range(train_count)
    ]
    rows.append(_span_row("development-1", "development", "Bệnh nhân sốt"))
    output_sha256 = write_jsonl(spans, rows)
    manifest = root / "human-manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "mined-span-dataset.v1",
            "chunk_count": len(rows),
            "entity_count": len(rows),
            "output_sha256": output_sha256,
            "augmentation": {
                "round2_included": False,
                "quarantined_data_included": False,
            },
        },
    )
    return spans, manifest


def _span_row(record_id: str, split: str, text: str) -> dict[str, object]:
    start = text.index("ho") if "ho" in text else text.index("sốt")
    end = start + (2 if "ho" in text else 3)
    digest = sha256_text(f"{record_id}\0{text}")
    return {
        "record_id": f"record:{record_id}",
        "document_id": f"document:{record_id}",
        "split": split,
        "text": text,
        "text_sha256": sha256_text(text),
        "source_span": [0, len(text)],
        "language": "vi",
        "note_type": "clinical",
        "source_artifact_id": "phase1-manual-gold:" + "a" * 64,
        "entities": [
            {
                "annotation_id": f"annotation:{digest[:16]}",
                "start": start,
                "end": end,
                "source_start": start,
                "source_end": end,
                "text": text[start:end],
                "label": "SYMPTOM",
            }
        ],
    }


def _synthetic_archive(
    root: Path,
    *,
    train_count: int,
    corrupt_test_offset: bool = False,
) -> Path:
    archive = root / "synthetic.zip"
    rows_by_split = {
        "train": [
            _source_row(
                f"train-{index:04d}",
                style=("qa_advice" if index % 2 == 0 else "structured_discharge"),
            )
            for index in range(train_count)
        ],
        "dev": [_source_row("dev-0001", style="qa_advice")],
        "test": [_source_row("test-0001", style="structured_discharge")],
    }
    if corrupt_test_offset:
        rows_by_split["test"][0]["entities"][0]["position"] = [0, 2]
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
        for split, rows in rows_by_split.items():
            content = "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows
            )
            output.writestr(
                f"viettel_medical_synthetic_v1/jsonl/{split}.jsonl",
                content,
            )
    return archive


def _source_row(record_id: str, *, style: str) -> dict[str, object]:
    text = f"{record_id}: bệnh nhân ho."
    start = text.index("ho")
    return {
        "id": record_id,
        "text": text,
        "style": style,
        "entities": [
            {
                "text": "ho",
                "type": "TRIỆU_CHỨNG",
                "candidates": [],
                "assertions": [],
                "position": [start, start + 2],
            }
        ],
    }
