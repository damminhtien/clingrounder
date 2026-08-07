"""Development-only calibration and explicit holdout gates for model NER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from clingrounder.benchmarks.phase1.model_selection import (
    Phase1ModelSelectionConfig,
    calibrate_phase1_model_thresholds,
    compare_phase1_ner_variants,
    infer_phase1_development_predictions,
    load_phase1_development_documents,
)
from clingrounder.pipeline.runner import PipelineRunner
from clingrounder.schema.annotation import EntityAnnotation
from clingrounder.schema.output import ClinicalPrediction
from clingrounder.schema.types import EntityType
from clingrounder.utils.hashing import sha256_file, sha256_text
from clingrounder.utils.text import normalize_for_match


def test_threshold_calibration_uses_only_development_and_removes_spurious(
    tmp_path: Path,
) -> None:
    config = _selection_fixture(tmp_path)
    predictions = {
        "1": _prediction(
            "1",
            "đau x",
            [
                _entity("M1", "đau", EntityType.SYMPTOM, 0.4, 0),
                _entity("M2", "x", EntityType.SYMPTOM, 0.3, 4),
            ],
        ),
        "2": _prediction(
            "2",
            "hen",
            [_entity("M1", "hen", EntityType.DISEASE, 0.9, 0)],
        ),
    }

    report = calibrate_phase1_model_thresholds(predictions, config=config)

    assert report["selection_split"] == "development"
    assert report["holdout_status"] == "sealed"
    assert report["selected_thresholds"]["SYMPTOM"] == 0.35
    assert report["selected_thresholds"]["DISEASE"] == 0.5
    assert report["error_counts"].get("phase1_spurious_entity", 0) == 0
    symptom = report["searches"]["SYMPTOM"]
    assert symptom["selection_objective"] == "phase1_score"
    assert symptom["stability"]["small_sample_warning"] is True
    assert symptom["stability"]["support"]["gold_entities"] == 1
    assert symptom["stability"]["grouped_repeated_cv"]["repeats"] == 5
    assert symptom["stability"]["bootstrap_95_ci"]["score"]["lower"] <= (
        symptom["stability"]["bootstrap_95_ci"]["score"]["upper"]
    )


def test_threshold_calibration_rejects_holdout_prediction(tmp_path: Path) -> None:
    config = _selection_fixture(tmp_path)
    predictions = {
        "1": _prediction("1", "đau x", []),
        "2": _prediction("2", "hen", []),
        "3": _prediction("3", "sealed", []),
    }

    with pytest.raises(ValueError, match="exactly the development predictions"):
        calibrate_phase1_model_thresholds(predictions, config=config)


def test_threshold_stability_keeps_duplicate_group_together(tmp_path: Path) -> None:
    config = _selection_fixture(tmp_path)
    split = json.loads(config.model_split_manifest.read_text(encoding="utf-8"))
    split["split_groups"] = {
        "phase1-manual-gold:1": "duplicate:same",
        "phase1-manual-gold:2": "duplicate:same",
    }
    config.model_split_manifest.write_text(
        json.dumps(split, sort_keys=True),
        encoding="utf-8",
    )
    predictions = {
        "1": _prediction(
            "1",
            "đau x",
            [_entity("M1", "đau", EntityType.SYMPTOM, 0.4, 0)],
        ),
        "2": _prediction(
            "2",
            "hen",
            [_entity("M1", "hen", EntityType.DISEASE, 0.9, 0)],
        ),
    }

    report = calibrate_phase1_model_thresholds(predictions, config=config)

    stability = report["searches"]["SYMPTOM"]["stability"]
    assert stability["support"]["group_count"] == 1
    assert stability["grouped_repeated_cv"] is None


def test_development_loader_never_opens_holdout_gold(tmp_path: Path) -> None:
    config = _selection_fixture(tmp_path)
    (config.gold_dir / "3.json").write_text("not-json", encoding="utf-8")

    documents = load_phase1_development_documents(config)

    assert [document.document_id for document in documents] == ["1", "2"]
    assert [document.text for document in documents] == ["đau x", "hen"]
    assert all(document.metadata == {"split": "development"} for document in documents)


def test_development_inference_uses_exact_loader_contract(tmp_path: Path) -> None:
    config = _selection_fixture(tmp_path)

    class FakeRunner:
        def process_document(self, document: object) -> ClinicalPrediction:
            document_id = str(getattr(document, "document_id"))
            text = str(getattr(document, "text"))
            return _prediction(document_id, text, [])

    predictions = infer_phase1_development_predictions(
        cast(PipelineRunner, FakeRunner()),
        config=config,
    )

    assert list(predictions) == ["1", "2"]
    assert predictions["1"].text_hash != predictions["2"].text_hash


def test_variant_comparison_keeps_holdout_sealed_until_explicitly_opened(
    tmp_path: Path,
) -> None:
    config = _selection_fixture(tmp_path)
    variants = {}
    for name, rows in {
        "rule": {"1": [], "2": []},
        "model": {
            "1": [_phase1_row("đau", "TRIỆU_CHỨNG", 0)],
            "2": [_phase1_row("hen", "CHẨN_ĐOÁN", 0)],
        },
        "hybrid": {
            "1": [
                _phase1_row("đau", "TRIỆU_CHỨNG", 0),
                _phase1_row("x", "TRIỆU_CHỨNG", 4),
            ],
            "2": [_phase1_row("hen", "CHẨN_ĐOÁN", 0)],
        },
    }.items():
        output = tmp_path / name
        output.mkdir()
        for document_id, document_rows in rows.items():
            (output / f"{document_id}.json").write_text(
                json.dumps(document_rows, ensure_ascii=False),
                encoding="utf-8",
            )
        variants[name] = output

    # A malformed holdout gold file proves the default comparison never opens it.
    (config.gold_dir / "3.json").write_text("not-json", encoding="utf-8")
    report = compare_phase1_ner_variants(variants, config=config)

    assert report["recommended_variant"] == "model"
    assert report["holdout_status"] == "sealed"
    assert all(row["holdout"] is None for row in report["variants"].values())


def test_variant_comparison_verifies_fingerprinted_holdout_baseline(
    tmp_path: Path,
) -> None:
    config = _selection_fixture(tmp_path)
    variants = _variant_fixture(tmp_path, include_holdout=True)

    report = compare_phase1_ner_variants(
        variants,
        config=config,
        open_frozen_holdout=True,
    )

    gate = report["variants"]["model"]["holdout"]["promotion_gate"]
    assert gate["baseline"]["artifact_id"] == "fixture-rule-baseline"
    assert len(gate["baseline"]["artifact_sha256"]) == 64


def test_variant_comparison_rejects_baseline_for_another_split(
    tmp_path: Path,
) -> None:
    config = _selection_fixture(tmp_path)
    variants = _variant_fixture(tmp_path, include_holdout=True)
    baseline = json.loads(config.holdout_baseline_artifact.read_text(encoding="utf-8"))
    baseline["contracts"]["holdout_document_ids_sha256"] = "0" * 64
    config.holdout_baseline_artifact.write_text(
        json.dumps(baseline, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="holdout baseline contract mismatch"):
        compare_phase1_ner_variants(
            variants,
            config=config,
            open_frozen_holdout=True,
        )


def _selection_fixture(tmp_path: Path) -> Phase1ModelSelectionConfig:
    input_dir = tmp_path / "input"
    gold_dir = tmp_path / "gold"
    input_dir.mkdir()
    gold_dir.mkdir()
    texts = {"1": "đau x", "2": "hen", "3": "sealed"}
    for document_id, text in texts.items():
        (input_dir / f"{document_id}.txt").write_text(text, encoding="utf-8")
    (gold_dir / "1.json").write_text(
        json.dumps([_phase1_row("đau", "TRIỆU_CHỨNG", 0)], ensure_ascii=False),
        encoding="utf-8",
    )
    (gold_dir / "2.json").write_text(
        json.dumps([_phase1_row("hen", "CHẨN_ĐOÁN", 0)], ensure_ascii=False),
        encoding="utf-8",
    )
    (gold_dir / "3.json").write_text("[]", encoding="utf-8")
    frozen = tmp_path / "holdout.json"
    frozen.write_text(
        json.dumps(
            {
                "corpus": {"fingerprint_sha256": "a" * 64},
                "splits": {
                    "train": {"document_ids": ["1", "2"]},
                    "holdout": {"document_ids": ["3"]},
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    model_split = tmp_path / "model-split.json"
    model_split.write_text(
        json.dumps(
            {
                "source_split_manifest_sha256": sha256_file(frozen),
                "source_document_ids": {
                    "train": [],
                    "development": ["1", "2"],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "holdout-baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "artifact_id": "fixture-rule-baseline",
                "schema_version": "phase1-ner-holdout-baseline.v1",
                "contracts": {
                    "frozen_split_manifest_sha256": sha256_file(frozen),
                    "model_split_manifest_sha256": sha256_file(model_split),
                    "corpus_fingerprint_sha256": "a" * 64,
                    "holdout_document_ids_sha256": sha256_text("3\n"),
                },
                "baseline": {
                    "metrics": {"score": 0.0, "text_score": 0.0},
                    "error_counts": {
                        "phase1_missing_entity": 1,
                        "phase1_spurious_entity": 0,
                        "phase1_text_boundary": 0,
                    },
                },
                "promotion_limits": {
                    "minimum_text_gain": 0.0,
                    "minimum_missing_reduction": 0,
                    "maximum_spurious": 10,
                    "maximum_boundary": 10,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return Phase1ModelSelectionConfig(
        input_dir=input_dir,
        gold_dir=gold_dir,
        model_split_manifest=model_split,
        frozen_split_manifest=frozen,
        holdout_baseline_artifact=baseline,
        threshold_grid=(0.0, 0.35, 0.5),
    )


def _variant_fixture(
    tmp_path: Path,
    *,
    include_holdout: bool,
) -> dict[str, Path]:
    rows_by_document = {
        "1": [_phase1_row("đau", "TRIỆU_CHỨNG", 0)],
        "2": [_phase1_row("hen", "CHẨN_ĐOÁN", 0)],
        "3": [],
    }
    variants: dict[str, Path] = {}
    for name in ("rule", "model", "hybrid"):
        output = tmp_path / f"opened-{name}"
        output.mkdir()
        document_ids = ("1", "2", "3") if include_holdout else ("1", "2")
        for document_id in document_ids:
            (output / f"{document_id}.json").write_text(
                json.dumps(rows_by_document[document_id], ensure_ascii=False),
                encoding="utf-8",
            )
        variants[name] = output
    return variants


def _entity(
    entity_id: str,
    text: str,
    entity_type: EntityType,
    confidence: float,
    start: int,
) -> EntityAnnotation:
    return EntityAnnotation(
        id=entity_id,
        span=(start, start + len(text)),
        text=text,
        normalized_text=normalize_for_match(text),
        type=entity_type,
        confidence=confidence,
    )


def _prediction(
    document_id: str,
    text: str,
    entities: list[EntityAnnotation],
) -> ClinicalPrediction:
    return ClinicalPrediction.from_text(
        document_id,
        text,
        entities,
        [],
        pipeline_version="test",
    )


def _phase1_row(text: str, entity_type: str, start: int) -> dict[str, object]:
    row: dict[str, object] = {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "position": [start, start + len(text)],
    }
    if entity_type in {"CHẨN_ĐOÁN", "THUỐC"}:
        row["candidates"] = []
    return row
