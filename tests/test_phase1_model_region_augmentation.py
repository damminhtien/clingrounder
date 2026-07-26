"""Q&A/educational augmentation must remain train-only and offset exact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.model_region_augmentation import (
    Phase1RegionAugmentationConfig,
    build_phase1_region_augmented_dataset,
)
from medical_kg_nlp.cli.main import main
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.training.span_dataset import (
    SpanTrainingRecord,
    scan_span_dataset,
    validate_span_dataset_manifest,
)
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text


def test_region_augmentation_is_deterministic_bounded_and_train_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_dataset, source_manifest, source_build, source_rows = _source_dataset(
        tmp_path / "source"
    )
    first_output = tmp_path / "augmented-a"
    second_output = tmp_path / "augmented-b"
    arguments = [
        "benchmark",
        "phase1",
        "model-data",
        "augment-regions",
        "--source-dataset",
        str(source_dataset),
        "--source-manifest",
        str(source_manifest),
        "--source-build-manifest",
        str(source_build),
        "--output-dir",
        str(first_output),
        "--max-synthetic-fraction",
        "0.4",
        "--seed",
        "test-region-seed",
    ]

    assert main(arguments) == 0
    first_report = json.loads(capsys.readouterr().out)
    config = Phase1RegionAugmentationConfig(
        source_dataset_path=source_dataset,
        source_manifest_path=source_manifest,
        source_build_manifest_path=source_build,
        max_synthetic_train_fraction=0.4,
        seed="test-region-seed",
    )
    assert build_phase1_region_augmented_dataset(first_output, config=config) == first_report
    second_report = build_phase1_region_augmented_dataset(second_output, config=config)

    rows = _jsonl(first_output / "spans.jsonl")
    synthetic = [
        row for row in rows if str(row["source_artifact_id"]).startswith("synthetic:")
    ]
    original_development = [
        row for row in source_rows if row["split"] == "development"
    ]
    augmented_development = [
        row for row in rows if row["split"] == "development"
    ]

    assert len(synthetic) == 2
    assert {row["note_type"] for row in synthetic} == {
        "question_answer",
        "educational",
    }
    assert augmented_development == original_development
    assert all(row["split"] == "train" for row in synthetic)
    assert {
        row["metadata"]["parent_record_id"] for row in synthetic
    }.issubset(
        {row["record_id"] for row in source_rows if row["split"] == "train"}
    )
    for row in synthetic:
        record = SpanTrainingRecord.from_mapping(row)
        for entity in record.entities:
            assert record.text[entity.start : entity.end] == entity.text
            assert entity.start == entity.source_start
            assert entity.end == entity.source_end

    summary = scan_span_dataset(first_output / "spans.jsonl")
    manifest = validate_span_dataset_manifest(
        first_output / "spans.jsonl",
        first_output / "manifest.json",
        summary,
    )
    assert summary.split_record_counts == {"development": 1, "train": 5}
    assert manifest["augmentation"]["synthetic_train_fraction"] == pytest.approx(0.4)
    assert manifest["augmentation"]["development_augmented"] is False
    assert manifest["augmentation"]["round2_included"] is False
    assert manifest["augmentation"]["quarantined_data_included"] is False
    assert first_report["outputs"]["spans.jsonl"] == second_report["outputs"][
        "spans.jsonl"
    ]


def test_region_augmentation_rejects_round2_or_quarantined_supervision(
    tmp_path: Path,
) -> None:
    source_dataset, source_manifest, source_build, _ = _source_dataset(
        tmp_path / "source",
        source_artifact_id="phase1_round2:private",
    )

    with pytest.raises(ValueError, match="Disallowed supervision source"):
        build_phase1_region_augmented_dataset(
            tmp_path / "output",
            config=Phase1RegionAugmentationConfig(
                source_dataset_path=source_dataset,
                source_manifest_path=source_manifest,
                source_build_manifest_path=source_build,
            ),
        )


def _source_dataset(
    directory: Path,
    *,
    source_artifact_id: str = "phase1-manual-gold:test",
) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    directory.mkdir(parents=True)
    rows = [
        _record(index, split="train", source_artifact_id=source_artifact_id)
        for index in range(3)
    ]
    rows.append(
        _record(3, split="development", source_artifact_id=source_artifact_id)
    )
    dataset = directory / "spans.jsonl"
    dataset_sha256 = write_jsonl(dataset, rows)
    entity_count = sum(len(row["entities"]) for row in rows)
    manifest = directory / "manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "mined-span-dataset.v1",
            "chunk_count": len(rows),
            "entity_count": entity_count,
            "output": "spans.jsonl",
            "output_sha256": dataset_sha256,
        },
    )
    build_manifest = directory / "build_manifest.json"
    write_json(
        build_manifest,
        {
            "schema_version": "phase1-five-type-model-dataset.v1",
            "build_contract": {
                "round2_included": False,
                "excluded_holdout_document_count": 1,
            },
            "outputs": {"spans.jsonl": sha256_file(dataset)},
        },
    )
    return dataset, manifest, build_manifest, rows


def _record(
    index: int,
    *,
    split: str,
    source_artifact_id: str,
) -> dict[str, object]:
    mention = f"triệu chứng được review {index}"
    text = f"Nội dung lâm sàng: {mention}."
    start = text.index(mention)
    end = start + len(mention)
    return {
        "record_id": f"source-record:{index}",
        "document_id": f"source-document:{index}",
        "split": split,
        "text": text,
        "text_sha256": sha256_text(text),
        "source_span": [0, len(text)],
        "language": "vi",
        "note_type": "clinical_note",
        "source_artifact_id": source_artifact_id,
        "entities": [
            {
                "annotation_id": f"source-annotation:{index}",
                "start": start,
                "end": end,
                "source_start": start,
                "source_end": end,
                "text": mention,
                "label": "SYMPTOM",
            }
        ],
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
