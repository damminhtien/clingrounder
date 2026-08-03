"""Tests for the provenance-checked final token training bundle."""

from __future__ import annotations

from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.final_token_training_bundle import (
    Phase1FinalTokenTrainingBundleConfig,
    build_phase1_final_token_training_bundle,
)
from medical_kg_nlp.cli.parser import build_parser
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.training.span_dataset import (
    scan_span_dataset,
    validate_span_dataset_manifest,
)
from medical_kg_nlp.utils.hashing import sha256_text


def test_final_bundle_keeps_authorized_rows_and_bounded_qa_augmentation(tmp_path: Path) -> None:
    final_dataset, final_manifest = _dataset(
        tmp_path / "final",
        rows=(_record("final-1", "phase1-final-supervision:pinned"), _record("final-2", "phase1-final-supervision:pinned")),
    )
    augmentation_dataset, augmentation_manifest = _dataset(
        tmp_path / "augmentation",
        rows=(
            _record("synthetic-1", "synthetic:phase1-region-renderer.v1", note_type="educational"),
            _record("manual-1", "phase1-manual-gold:pinned", note_type="clinical"),
        ),
    )
    output = tmp_path / "bundle"

    config = Phase1FinalTokenTrainingBundleConfig(
        final_dataset_path=final_dataset,
        final_manifest_path=final_manifest,
        augmentation_dataset_path=augmentation_dataset,
        augmentation_manifest_path=augmentation_manifest,
        output_dir=output,
    )
    first = build_phase1_final_token_training_bundle(config)
    repeated = build_phase1_final_token_training_bundle(config)

    summary = scan_span_dataset(output / "spans.jsonl")
    manifest = validate_span_dataset_manifest(output / "spans.jsonl", output / "manifest.json", summary)
    assert summary.record_count == 3
    assert manifest["entity_type_counts"] == {"SYMPTOM": 3}
    assert first["augmentation"]["selected_records"] == 1
    assert first["augmentation"]["selected_fraction"] == pytest.approx(1 / 3)
    assert first == repeated


def test_final_bundle_rejects_non_final_or_round2_provenance(tmp_path: Path) -> None:
    final_dataset, final_manifest = _dataset(
        tmp_path / "final",
        rows=(_record("invalid-final", "phase1_round2:private"),),
        config={"round2_included": True},
    )
    augmentation_dataset, augmentation_manifest = _dataset(
        tmp_path / "augmentation",
        rows=(_record("synthetic-1", "synthetic:phase1-region-renderer.v1", note_type="question_answer"),),
    )

    with pytest.raises(ValueError, match="disallowed provenance"):
        build_phase1_final_token_training_bundle(
            Phase1FinalTokenTrainingBundleConfig(
                final_dataset_path=final_dataset,
                final_manifest_path=final_manifest,
                augmentation_dataset_path=augmentation_dataset,
                augmentation_manifest_path=augmentation_manifest,
                output_dir=tmp_path / "bundle",
            )
        )


def test_final_bundle_cli_exposes_only_pinned_dataset_inputs() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "phase1",
            "model-data",
            "build-final-fit-bundle",
            "--output-dir",
            "outputs/models/final-bundle",
        ]
    )

    assert args.handler == "benchmark_phase1_model_data_build_final_fit_bundle"
    assert args.maximum_synthetic_fraction == pytest.approx(0.4)
    assert args.final_dataset.endswith("phase1-final-supervision-five-type-v1/spans.jsonl")


def _dataset(
    directory: Path,
    *,
    rows: tuple[dict[str, object], ...],
    config: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True)
    dataset = directory / "spans.jsonl"
    dataset_sha256 = write_jsonl(dataset, rows)
    manifest = directory / "manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "mined-span-dataset.v1",
            "chunk_count": len(rows),
            "entity_count": len(rows),
            "output": "spans.jsonl",
            "output_sha256": dataset_sha256,
            "config": config or {"round2_included": False},
            "augmentation": {"quarantined_data_included": False},
        },
    )
    return dataset, manifest


def _record(
    identifier: str,
    source_artifact_id: str,
    *,
    note_type: str = "phase1_final_supervision",
) -> dict[str, object]:
    text = "Bệnh nhân khó thở."
    start = text.index("khó thở")
    end = start + len("khó thở")
    return {
        "record_id": identifier,
        "document_id": identifier,
        "split": "train",
        "text": text,
        "text_sha256": sha256_text(text),
        "source_span": [0, len(text)],
        "language": "vi",
        "note_type": note_type,
        "source_artifact_id": source_artifact_id,
        "entities": [
            {
                "annotation_id": f"annotation-{identifier}",
                "start": start,
                "end": end,
                "source_start": start,
                "source_end": end,
                "text": "khó thở",
                "label": "SYMPTOM",
            }
        ],
    }
