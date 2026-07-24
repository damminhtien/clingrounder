"""Leakage-safe threshold calibration and NER variant comparison for Phase 1.

The generic token classifier emits raw-offset entities and confidences. This benchmark module
owns the task-specific label mapping, frozen split contracts, and promotion thresholds. Keeping
those decisions here prevents Phase 1 scoring conventions from leaking into reusable adapters.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.phase1 import (
    prediction_to_phase1_entities,
    score_phase1_documents,
)
from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import load_phase1_output_source
from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.ontology.phase1 import PHASE1_ENTITY_TYPE_RULES
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.schema.validator import prediction_from_json
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text
from medical_kg_nlp.utils.io import read_jsonl, read_source_text

__all__ = [
    "PHASE1_NER_VARIANTS",
    "Phase1HoldoutGate",
    "Phase1ModelSelectionConfig",
    "calibrate_phase1_model_thresholds",
    "compare_phase1_ner_variants",
    "load_phase1_development_documents",
    "load_internal_predictions",
    "write_phase1_model_selection_report",
]

PHASE1_NER_VARIANTS = ("rule", "model", "hybrid")
_PHASE1_TYPE_BY_INTERNAL = {
    rule.internal_type: rule.phase1_type for rule in PHASE1_ENTITY_TYPE_RULES
}


@dataclass(frozen=True)
class Phase1HoldoutGate:
    """Frozen rule baseline and promotion limits from the complete manual-gold audit."""

    score: float = 53.039409
    text_score: float = 0.530395
    missing: int = 235
    spurious: int = 38
    boundary: int = 107
    minimum_text_gain: float = 0.015
    minimum_missing_reduction: int = 20
    maximum_spurious: int = 43
    maximum_boundary: int = 112


@dataclass(frozen=True)
class Phase1ModelSelectionConfig:
    """Immutable inputs used for development calibration and optional holdout opening."""

    input_dir: Path = Path("data/raw/input")
    gold_dir: Path = Path("data/manual_gold")
    model_split_manifest: Path = Path(
        "outputs/mining/model-datasets/phase1-manual-five-type-v1/split_manifest.json"
    )
    frozen_split_manifest: Path = Path("data/manual_gold/holdout_manifest.json")
    threshold_grid: tuple[float, ...] = (
        0.0,
        0.25,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
    )
    holdout_gate: Phase1HoldoutGate = Phase1HoldoutGate()

    def __post_init__(self) -> None:
        if not self.threshold_grid:
            raise ValueError("threshold_grid must contain at least one value")
        if any(not 0.0 <= value <= 1.0 for value in self.threshold_grid):
            raise ValueError("threshold_grid values must be between zero and one")
        if tuple(sorted(set(self.threshold_grid))) != self.threshold_grid:
            raise ValueError("threshold_grid must be strictly increasing")


def load_internal_predictions(path: str | Path) -> dict[str, ClinicalPrediction]:
    """Load typed internal predictions while rejecting duplicate document IDs."""

    predictions: dict[str, ClinicalPrediction] = {}
    for payload in read_jsonl(path):
        prediction = prediction_from_json(payload)
        if prediction.document_id in predictions:
            raise ValueError(
                f"Duplicate internal prediction for document {prediction.document_id!r}"
            )
        predictions[prediction.document_id] = prediction
    return predictions


def load_phase1_development_documents(
    config: Phase1ModelSelectionConfig | None = None,
) -> tuple[ClinicalDocument, ...]:
    """Load only development inputs, without opening any gold or holdout file."""

    active = config or Phase1ModelSelectionConfig()
    development_ids = _load_split_contracts(active)["development_ids"]
    source_texts = _load_source_texts(active.input_dir, development_ids)
    # INVARIANT: inference sees exactly the IDs accepted later by threshold calibration.
    return tuple(
        ClinicalDocument(
            document_id=document_id,
            text=source_texts[document_id],
            metadata={"split": "development"},
        )
        for document_id in development_ids
    )


def calibrate_phase1_model_thresholds(
    predictions: Mapping[str, ClinicalPrediction],
    *,
    config: Phase1ModelSelectionConfig | None = None,
    prediction_path: str | Path | None = None,
) -> dict[str, Any]:
    """Choose independent per-type thresholds using only the model development split.

    The calibration input must contain exactly the development document IDs. Requiring a
    development-only inference artifact is deliberate: it makes accidental holdout scoring
    impossible even if a caller later refactors loading code.
    """

    active = config or Phase1ModelSelectionConfig()
    contracts = _load_split_contracts(active)
    development_ids = contracts["development_ids"]
    actual_ids = set(predictions)
    if actual_ids != set(development_ids):
        raise ValueError(
            "Threshold calibration requires exactly the development predictions: "
            f"missing={_sort_ids(set(development_ids) - actual_ids)}, "
            f"unexpected={_sort_ids(actual_ids - set(development_ids))}"
        )

    source_texts = _load_source_texts(active.input_dir, development_ids)
    gold = _load_gold_rows(active.gold_dir, development_ids)
    _validate_predictions(predictions, source_texts)

    selected: dict[EntityType, float] = {}
    searches: dict[str, Any] = {}
    for entity_type in _PHASE1_TYPE_BY_INTERNAL:
        phase1_type = _PHASE1_TYPE_BY_INTERNAL[entity_type]
        typed_gold = {
            document_id: [
                row for row in rows if str(row.get("type")) == phase1_type
            ]
            for document_id, rows in gold.items()
        }
        trials: list[dict[str, Any]] = []
        for threshold in active.threshold_grid:
            rows = _prediction_rows(
                predictions,
                source_texts,
                thresholds={entity_type: threshold},
                include_types=frozenset({entity_type}),
            )
            metrics, errors = score_phase1_documents(typed_gold, rows)
            trials.append(_trial(threshold, metrics, errors))
        best = max(
            trials,
            key=lambda row: (
                float(row["text_score"]),
                -int(row["spurious"]),
                -int(row["boundary"]),
                -int(row["missing"]),
                float(row["threshold"]),
            ),
        )
        selected[entity_type] = float(best["threshold"])
        searches[entity_type.value] = {
            "phase1_type": phase1_type,
            "selected_threshold": best["threshold"],
            "trials": trials,
        }

    calibrated_rows = _prediction_rows(
        predictions,
        source_texts,
        thresholds=selected,
        include_types=frozenset(selected),
    )
    metrics, errors = score_phase1_documents(gold, calibrated_rows)
    return {
        "schema_version": "phase1-model-threshold-calibration.v1",
        "selection_split": "development",
        "holdout_status": "sealed",
        "document_count": len(development_ids),
        "document_ids_sha256": _ids_sha256(development_ids),
        "selected_thresholds": {
            entity_type.value: selected[entity_type]
            for entity_type in sorted(selected, key=lambda value: value.value)
        },
        "metrics": metrics,
        "error_counts": _error_counts(errors),
        "searches": searches,
        "inputs": _input_fingerprints(active, prediction_path=prediction_path),
    }


def compare_phase1_ner_variants(
    variants: Mapping[str, str | Path],
    *,
    config: Phase1ModelSelectionConfig | None = None,
    open_frozen_holdout: bool = False,
) -> dict[str, Any]:
    """Compare rule/model/hybrid flat outputs on development, then optionally holdout."""

    active = config or Phase1ModelSelectionConfig()
    missing_variants = set(PHASE1_NER_VARIANTS) - set(variants)
    extra_variants = set(variants) - set(PHASE1_NER_VARIANTS)
    if missing_variants or extra_variants:
        raise ValueError(
            "Variants must be exactly rule, model, and hybrid: "
            f"missing={sorted(missing_variants)}, extra={sorted(extra_variants)}"
        )
    contracts = _load_split_contracts(active)
    development_ids = contracts["development_ids"]
    development_gold = _load_gold_rows(active.gold_dir, development_ids)
    loaded = {
        name: load_phase1_output_source(path) for name, path in variants.items()
    }
    reports: dict[str, Any] = {}
    for name in PHASE1_NER_VARIANTS:
        reports[name] = {
            "development": _score_subset(
                development_gold,
                loaded[name],
                development_ids,
            ),
            "holdout": None,
            "input": {
                "path": str(variants[name]),
                "sha256": _path_sha256(Path(variants[name])),
            },
        }

    ranked = sorted(
        PHASE1_NER_VARIANTS,
        key=lambda name: (
            reports[name]["development"]["metrics"]["text_score"],
            reports[name]["development"]["metrics"]["score"],
            -reports[name]["development"]["error_counts"].get(
                "phase1_spurious_entity", 0
            ),
            name,
        ),
        reverse=True,
    )

    if open_frozen_holdout:
        holdout_ids = contracts["holdout_ids"]
        holdout_gold = _load_gold_rows(active.gold_dir, holdout_ids)
        for name in PHASE1_NER_VARIANTS:
            scored = _score_subset(holdout_gold, loaded[name], holdout_ids)
            scored["promotion_gate"] = _holdout_gate(
                scored,
                active.holdout_gate,
            )
            reports[name]["holdout"] = scored

    return {
        "schema_version": "phase1-ner-variant-comparison.v1",
        "selection_split": "development",
        "holdout_status": "opened_for_final_gate" if open_frozen_holdout else "sealed",
        "ranking": ranked,
        "recommended_variant": ranked[0],
        "variants": reports,
        "inputs": _input_fingerprints(active),
    }


def write_phase1_model_selection_report(
    report: Mapping[str, Any], path: str | Path
) -> None:
    """Persist one deterministic calibration or comparison report."""

    write_json(path, report)


def _load_split_contracts(config: Phase1ModelSelectionConfig) -> dict[str, tuple[str, ...]]:
    model = _read_mapping(config.model_split_manifest)
    frozen = _read_mapping(config.frozen_split_manifest)
    expected_frozen_hash = str(model.get("source_split_manifest_sha256", ""))
    actual_frozen_hash = sha256_file(config.frozen_split_manifest)
    if expected_frozen_hash != actual_frozen_hash:
        raise ValueError(
            "Model split does not reference the current frozen holdout manifest"
        )
    source_ids = model.get("source_document_ids")
    if not isinstance(source_ids, Mapping):
        raise ValueError("Model split manifest is missing source_document_ids")
    development_ids = _string_ids(source_ids.get("development"), "development")
    frozen_splits = frozen.get("splits")
    if not isinstance(frozen_splits, Mapping):
        raise ValueError("Frozen split manifest is missing splits")
    holdout = frozen_splits.get("holdout")
    if not isinstance(holdout, Mapping):
        raise ValueError("Frozen split manifest is missing holdout")
    holdout_ids = _string_ids(holdout.get("document_ids"), "holdout")
    if set(development_ids) & set(holdout_ids):
        raise ValueError("Model development split overlaps the frozen holdout")
    return {
        "development_ids": development_ids,
        "holdout_ids": holdout_ids,
    }


def _prediction_rows(
    predictions: Mapping[str, ClinicalPrediction],
    source_texts: Mapping[str, str],
    *,
    thresholds: Mapping[EntityType, float],
    include_types: frozenset[EntityType],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for document_id in _sort_ids(set(predictions)):
        prediction = predictions[document_id]
        entities = [
            entity
            for entity in prediction.entities
            if entity.type in include_types
            and entity.confidence >= thresholds.get(entity.type, 0.0)
        ]
        filtered = replace(prediction, entities=entities, relations=[])
        rows[document_id] = prediction_to_phase1_entities(
            filtered,
            source_text=source_texts[document_id],
            assertion_policy="empty",
            candidate_policy="empty",
        )
    return rows


def _validate_predictions(
    predictions: Mapping[str, ClinicalPrediction], source_texts: Mapping[str, str]
) -> None:
    for document_id, prediction in predictions.items():
        source_text = source_texts[document_id]
        if prediction.text_hash != sha256_text(source_text):
            raise ValueError(f"Prediction text hash mismatch for document {document_id}")
        prediction.validate(source_text)


def _load_source_texts(input_dir: Path, document_ids: Sequence[str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for document_id in document_ids:
        path = input_dir / f"{document_id}.txt"
        if not path.is_file():
            raise ValueError(f"Missing Phase 1 source document: {path}")
        rows[document_id] = read_source_text(path)
    return rows


def _load_gold_rows(
    gold_dir: Path, document_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    # INVARIANT: open only selected gold files; calibration never deserializes holdout labels.
    for document_id in document_ids:
        path = gold_dir / f"{document_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise ValueError(f"{path}: expected a JSON list of objects")
        rows[document_id] = [dict(row) for row in payload]
    return rows


def _score_subset(
    gold: dict[str, list[dict[str, Any]]],
    predictions: Mapping[str, list[dict[str, Any]]],
    document_ids: Sequence[str],
) -> dict[str, Any]:
    missing = set(document_ids) - set(predictions)
    if missing:
        raise ValueError(f"Variant is missing documents: {_sort_ids(missing)}")
    selected = {document_id: predictions[document_id] for document_id in document_ids}
    metrics, errors = score_phase1_documents(gold, selected)
    return {
        "document_count": len(document_ids),
        "metrics": metrics,
        "error_counts": _error_counts(errors),
    }


def _holdout_gate(report: Mapping[str, Any], gate: Phase1HoldoutGate) -> dict[str, Any]:
    metrics = report["metrics"]
    errors = report["error_counts"]
    text_gain = float(metrics["text_score"]) - gate.text_score
    missing_reduction = gate.missing - int(errors.get("phase1_missing_entity", 0))
    checks = {
        "overall_score_non_decreasing": float(metrics["score"]) >= gate.score,
        "minimum_text_gain": text_gain >= gate.minimum_text_gain,
        "minimum_missing_reduction": missing_reduction >= gate.minimum_missing_reduction,
        "maximum_spurious": int(errors.get("phase1_spurious_entity", 0))
        <= gate.maximum_spurious,
        "maximum_boundary": int(errors.get("phase1_text_boundary", 0))
        <= gate.maximum_boundary,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": {
            "score": round(float(metrics["score"]) - gate.score, 6),
            "text_score": round(text_gain, 6),
            "missing_reduction": missing_reduction,
            "spurious_delta": int(errors.get("phase1_spurious_entity", 0))
            - gate.spurious,
            "boundary_delta": int(errors.get("phase1_text_boundary", 0))
            - gate.boundary,
        },
    }


def _trial(
    threshold: float,
    metrics: Mapping[str, Any],
    errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = _error_counts(errors)
    return {
        "threshold": threshold,
        "text_score": metrics["text_score"],
        "score": metrics["score"],
        "predicted_entities": metrics["predicted_entities"],
        "missing": counts.get("phase1_missing_entity", 0),
        "spurious": counts.get("phase1_spurious_entity", 0),
        "boundary": counts.get("phase1_text_boundary", 0),
    }


def _error_counts(errors: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["error_type"]) for row in errors).items()))


def _input_fingerprints(
    config: Phase1ModelSelectionConfig,
    *,
    prediction_path: str | Path | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "model_split_manifest": {
            "path": str(config.model_split_manifest),
            "sha256": sha256_file(config.model_split_manifest),
        },
        "frozen_split_manifest": {
            "path": str(config.frozen_split_manifest),
            "sha256": sha256_file(config.frozen_split_manifest),
        },
        "threshold_grid": list(config.threshold_grid),
    }
    if prediction_path is not None:
        path = Path(prediction_path)
        values["predictions"] = {"path": str(path), "sha256": sha256_file(path)}
    return values


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _string_ids(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} document IDs must be a list of strings")
    result = tuple(_sort_ids(set(value)))
    if len(result) != len(value):
        raise ValueError(f"{label} document IDs contain duplicates")
    return result


def _ids_sha256(document_ids: Sequence[str]) -> str:
    return sha256_text("\n".join(document_ids) + "\n")


def _sort_ids(values: set[str] | Sequence[str]) -> list[str]:
    return sorted(values, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def _path_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"Variant path does not exist: {path}")
    digest_rows = [
        f"{child.relative_to(path).as_posix()}\0{sha256_file(child)}"
        for child in sorted(value for value in path.rglob("*") if value.is_file())
    ]
    return sha256_text("\n".join(digest_rows) + "\n")
