"""Leakage-safe threshold calibration and NER variant comparison for Phase 1.

The generic token classifier emits raw-offset entities and confidences. This benchmark module
owns the task-specific label mapping, frozen split contracts, and promotion thresholds. Keeping
those decisions here prevents Phase 1 scoring conventions from leaking into reusable adapters.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypedDict

from medical_kg_nlp.benchmarks.phase1.phase1 import (
    prediction_to_phase1_entities,
    score_phase1_documents,
)
from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import load_phase1_output_source
from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.ontology.phase1 import PHASE1_ENTITY_TYPE_RULES
from medical_kg_nlp.pipeline.runner import PipelineRunner
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
    "infer_phase1_development_predictions",
    "load_phase1_development_documents",
    "load_internal_predictions",
    "write_phase1_model_selection_report",
]

PHASE1_NER_VARIANTS = ("rule", "model", "hybrid")
_PHASE1_TYPE_BY_INTERNAL = {
    rule.internal_type: rule.phase1_type for rule in PHASE1_ENTITY_TYPE_RULES
}
_CALIBRATION_CV_REPEATS = 5
_CALIBRATION_CV_FOLDS = 4
_CALIBRATION_BOOTSTRAP_REPLICATES = 200


class _SplitContracts(TypedDict):
    development_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    development_groups: dict[str, str]


@dataclass(frozen=True)
class Phase1HoldoutGate:
    """Verified baseline values loaded from one fingerprinted holdout artifact."""

    artifact_id: str
    artifact_sha256: str
    score: float
    text_score: float
    missing: int
    spurious: int
    boundary: int
    minimum_text_gain: float
    minimum_missing_reduction: int
    maximum_spurious: int
    maximum_boundary: int


@dataclass(frozen=True)
class Phase1ModelSelectionConfig:
    """Immutable inputs used for development calibration and optional holdout opening."""

    input_dir: Path = Path("data/raw/input")
    gold_dir: Path = Path("data/manual_gold")
    model_split_manifest: Path = Path(
        "outputs/mining/model-datasets/phase1-manual-five-type-v1/split_manifest.json"
    )
    frozen_split_manifest: Path = Path("data/manual_gold/holdout_manifest.json")
    holdout_baseline_artifact: Path = Path(
        "data/manual_gold/model_holdout_baseline.json"
    )
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


def infer_phase1_development_predictions(
    runner: PipelineRunner,
    *,
    config: Phase1ModelSelectionConfig | None = None,
) -> dict[str, ClinicalPrediction]:
    """Run one already-composed model over exactly the development documents."""

    predictions: dict[str, ClinicalPrediction] = {}
    for document in load_phase1_development_documents(config):
        prediction = runner.process_document(document)
        if prediction.document_id in predictions:
            raise ValueError(
                f"Duplicate development prediction for {prediction.document_id!r}"
            )
        predictions[prediction.document_id] = prediction
    return predictions


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
            key=_trial_order,
        )
        selected[entity_type] = float(best["threshold"])
        searches[entity_type.value] = {
            "phase1_type": phase1_type,
            "selected_threshold": best["threshold"],
            "selection_objective": "phase1_score",
            "stability": _threshold_stability(
                predictions,
                source_texts,
                typed_gold,
                entity_type=entity_type,
                selected_threshold=float(best["threshold"]),
                threshold_grid=active.threshold_grid,
                groups=contracts["development_groups"],
            ),
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
        "schema_version": "phase1-model-threshold-calibration.v2",
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
        holdout_gate = _load_holdout_gate(active, holdout_ids)
        for name in PHASE1_NER_VARIANTS:
            scored = _score_subset(holdout_gold, loaded[name], holdout_ids)
            scored["promotion_gate"] = _holdout_gate(
                scored,
                holdout_gate,
            )
            reports[name]["holdout"] = scored

    return {
        "schema_version": "phase1-ner-variant-comparison.v2",
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


def _load_split_contracts(config: Phase1ModelSelectionConfig) -> _SplitContracts:
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
    development_groups = _development_groups(model, development_ids)
    return {
        "development_ids": development_ids,
        "holdout_ids": holdout_ids,
        "development_groups": development_groups,
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
        "baseline": {
            "artifact_id": gate.artifact_id,
            "artifact_sha256": gate.artifact_sha256,
        },
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


def _trial_order(row: Mapping[str, Any]) -> tuple[float, float, int, int, int, float]:
    """Rank thresholds by the official objective, then deterministic error tie-breaks."""

    return (
        float(row["score"]),
        float(row["text_score"]),
        -int(row["spurious"]),
        -int(row["boundary"]),
        -int(row["missing"]),
        float(row["threshold"]),
    )


def _threshold_stability(
    predictions: Mapping[str, ClinicalPrediction],
    source_texts: Mapping[str, str],
    typed_gold: Mapping[str, list[dict[str, Any]]],
    *,
    entity_type: EntityType,
    selected_threshold: float,
    threshold_grid: Sequence[float],
    groups: Mapping[str, str],
) -> dict[str, Any]:
    """Report small-sample sensitivity without opening holdout labels.

    Grouped repeated CV estimates threshold selection stability. A deterministic document
    bootstrap reports uncertainty for the final all-development threshold. Both are diagnostics:
    they do not manufacture extra supervision or silently relax the promotion gate.
    """

    document_ids = tuple(_sort_ids(set(typed_gold)))
    gold_entity_count = sum(len(rows) for rows in typed_gold.values())
    predicted_entity_count = sum(
        1
        for prediction in predictions.values()
        for entity in prediction.entities
        if entity.type == entity_type
    )
    support = {
        "document_count": len(document_ids),
        "documents_with_gold": sum(bool(rows) for rows in typed_gold.values()),
        "gold_entities": gold_entity_count,
        "predicted_entities": predicted_entity_count,
        "group_count": len({groups[document_id] for document_id in document_ids}),
    }
    if not document_ids or gold_entity_count == 0:
        return {
            "status": "insufficient_gold",
            "support": support,
            "grouped_repeated_cv": None,
            "bootstrap_95_ci": None,
        }

    cross_validation = _grouped_threshold_cross_validation(
        predictions,
        source_texts,
        typed_gold,
        entity_type=entity_type,
        threshold_grid=threshold_grid,
        groups=groups,
    )
    bootstrap = _bootstrap_threshold_metrics(
        predictions,
        source_texts,
        typed_gold,
        entity_type=entity_type,
        threshold=selected_threshold,
        seed_material=f"{entity_type.value}:{_ids_sha256(document_ids)}",
    )
    return {
        "status": "diagnostic_only",
        "small_sample_warning": len(document_ids) < 30,
        "support": support,
        "grouped_repeated_cv": cross_validation,
        "bootstrap_95_ci": bootstrap,
    }


def _grouped_threshold_cross_validation(
    predictions: Mapping[str, ClinicalPrediction],
    source_texts: Mapping[str, str],
    typed_gold: Mapping[str, list[dict[str, Any]]],
    *,
    entity_type: EntityType,
    threshold_grid: Sequence[float],
    groups: Mapping[str, str],
) -> dict[str, Any] | None:
    unique_groups = sorted(set(groups.values()))
    fold_count = min(_CALIBRATION_CV_FOLDS, len(unique_groups))
    if fold_count < 2:
        return None

    folds: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for repeat in range(_CALIBRATION_CV_REPEATS):
        fold_by_group = {
            group: int(sha256_text(f"{repeat}:{group}")[:8], 16) % fold_count
            for group in unique_groups
        }
        for fold in range(fold_count):
            held_out = tuple(
                document_id
                for document_id in typed_gold
                if fold_by_group[groups[document_id]] == fold
            )
            train = tuple(document_id for document_id in typed_gold if document_id not in held_out)
            if not train or not held_out:
                continue
            trials = [
                _score_type_threshold(
                    predictions,
                    source_texts,
                    typed_gold,
                    entity_type=entity_type,
                    threshold=threshold,
                    document_ids=train,
                )
                for threshold in threshold_grid
            ]
            selected = max(trials, key=_trial_order)
            threshold = float(selected["threshold"])
            selected_counts[f"{threshold:.6f}"] += 1
            held_metrics = _score_type_threshold(
                predictions,
                source_texts,
                typed_gold,
                entity_type=entity_type,
                threshold=threshold,
                document_ids=held_out,
            )
            folds.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_documents": len(train),
                    "held_out_documents": len(held_out),
                    "selected_threshold": threshold,
                    "held_out_score": held_metrics["score"],
                    "held_out_text_score": held_metrics["text_score"],
                }
            )
    return {
        "repeats": _CALIBRATION_CV_REPEATS,
        "folds_per_repeat": fold_count,
        "evaluated_fold_count": len(folds),
        "selected_threshold_counts": dict(sorted(selected_counts.items())),
        "mean_held_out_score": _mean(row["held_out_score"] for row in folds),
        "mean_held_out_text_score": _mean(row["held_out_text_score"] for row in folds),
        "fold_metrics": folds,
    }


def _bootstrap_threshold_metrics(
    predictions: Mapping[str, ClinicalPrediction],
    source_texts: Mapping[str, str],
    typed_gold: Mapping[str, list[dict[str, Any]]],
    *,
    entity_type: EntityType,
    threshold: float,
    seed_material: str,
) -> dict[str, Any]:
    document_ids = tuple(_sort_ids(set(typed_gold)))
    rng = random.Random(int(sha256_text(seed_material)[:16], 16))
    scores: list[float] = []
    text_scores: list[float] = []
    rows_by_document = _prediction_rows(
        predictions,
        source_texts,
        thresholds={entity_type: threshold},
        include_types=frozenset({entity_type}),
    )
    for _ in range(_CALIBRATION_BOOTSTRAP_REPLICATES):
        sampled = tuple(rng.choice(document_ids) for _ in document_ids)
        sampled_gold: dict[str, list[dict[str, Any]]] = {}
        sampled_rows: dict[str, list[dict[str, Any]]] = {}
        for index, document_id in enumerate(sampled):
            sample_id = f"{index}:{document_id}"
            sampled_gold[sample_id] = typed_gold[document_id]
            sampled_rows[sample_id] = rows_by_document[document_id]
        metrics, _ = score_phase1_documents(sampled_gold, sampled_rows)
        scores.append(float(metrics["score"]))
        text_scores.append(float(metrics["text_score"]))
    return {
        "replicates": _CALIBRATION_BOOTSTRAP_REPLICATES,
        "score": _percentile_interval(scores),
        "text_score": _percentile_interval(text_scores),
    }


def _score_type_threshold(
    predictions: Mapping[str, ClinicalPrediction],
    source_texts: Mapping[str, str],
    typed_gold: Mapping[str, list[dict[str, Any]]],
    *,
    entity_type: EntityType,
    threshold: float,
    document_ids: Sequence[str],
) -> dict[str, Any]:
    selected_predictions = {
        document_id: predictions[document_id] for document_id in document_ids
    }
    selected_texts = {
        document_id: source_texts[document_id] for document_id in document_ids
    }
    rows = _prediction_rows(
        selected_predictions,
        selected_texts,
        thresholds={entity_type: threshold},
        include_types=frozenset({entity_type}),
    )
    metrics, errors = score_phase1_documents(
        {document_id: typed_gold[document_id] for document_id in document_ids},
        rows,
    )
    return _trial(threshold, metrics, errors)


def _percentile_interval(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return {
        "mean": round(sum(ordered) / len(ordered), 6),
        "lower": round(lower, 6),
        "upper": round(upper, 6),
    }


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return round(sum(rows) / len(rows), 6) if rows else 0.0


def _error_counts(errors: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["error_type"]) for row in errors).items()))


def _development_groups(
    model_manifest: Mapping[str, Any],
    development_ids: Sequence[str],
) -> dict[str, str]:
    raw_groups = model_manifest.get("split_groups")
    if not isinstance(raw_groups, Mapping):
        return {document_id: f"document:{document_id}" for document_id in development_ids}
    groups: dict[str, str] = {}
    for document_id in development_ids:
        value = raw_groups.get(document_id)
        if value is None:
            value = raw_groups.get(f"phase1-manual-gold:{document_id}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Model split has no duplicate group for document {document_id}")
        groups[document_id] = value
    return groups


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
        "holdout_baseline_artifact": {
            "path": str(config.holdout_baseline_artifact),
            "sha256": sha256_file(config.holdout_baseline_artifact),
        },
        "threshold_grid": list(config.threshold_grid),
    }
    if prediction_path is not None:
        path = Path(prediction_path)
        values["predictions"] = {"path": str(path), "sha256": sha256_file(path)}
    return values


def _load_holdout_gate(
    config: Phase1ModelSelectionConfig,
    holdout_ids: Sequence[str],
) -> Phase1HoldoutGate:
    """Load a gate only after verifying every dataset contract it claims to score."""

    path = config.holdout_baseline_artifact
    payload = _read_mapping(path)
    if payload.get("schema_version") != "phase1-ner-holdout-baseline.v1":
        raise ValueError(f"{path}: unsupported holdout baseline schema")

    contracts = payload.get("contracts")
    baseline = payload.get("baseline")
    limits = payload.get("promotion_limits")
    if not all(isinstance(value, Mapping) for value in (contracts, baseline, limits)):
        raise ValueError(f"{path}: missing contracts, baseline, or promotion_limits")
    assert isinstance(contracts, Mapping)
    assert isinstance(baseline, Mapping)
    assert isinstance(limits, Mapping)

    frozen = _read_mapping(config.frozen_split_manifest)
    corpus = frozen.get("corpus")
    if not isinstance(corpus, Mapping):
        raise ValueError("Frozen split manifest is missing its corpus fingerprint")
    expected_contracts = {
        "frozen_split_manifest_sha256": sha256_file(config.frozen_split_manifest),
        "model_split_manifest_sha256": sha256_file(config.model_split_manifest),
        "corpus_fingerprint_sha256": str(corpus.get("fingerprint_sha256", "")),
        "holdout_document_ids_sha256": _ids_sha256(holdout_ids),
    }
    mismatches = {
        key: {"expected": expected, "actual": contracts.get(key)}
        for key, expected in expected_contracts.items()
        if contracts.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{path}: holdout baseline contract mismatch: {mismatches}")

    metrics = baseline.get("metrics")
    errors = baseline.get("error_counts")
    if not isinstance(metrics, Mapping) or not isinstance(errors, Mapping):
        raise ValueError(f"{path}: baseline metrics or error_counts are missing")
    return Phase1HoldoutGate(
        artifact_id=_required_string(payload.get("artifact_id"), "artifact_id"),
        artifact_sha256=sha256_file(path),
        score=_required_float(metrics.get("score"), "baseline.metrics.score"),
        text_score=_required_float(
            metrics.get("text_score"), "baseline.metrics.text_score"
        ),
        missing=_required_int(
            errors.get("phase1_missing_entity"),
            "baseline.error_counts.phase1_missing_entity",
        ),
        spurious=_required_int(
            errors.get("phase1_spurious_entity"),
            "baseline.error_counts.phase1_spurious_entity",
        ),
        boundary=_required_int(
            errors.get("phase1_text_boundary"),
            "baseline.error_counts.phase1_text_boundary",
        ),
        minimum_text_gain=_required_float(
            limits.get("minimum_text_gain"),
            "promotion_limits.minimum_text_gain",
        ),
        minimum_missing_reduction=_required_int(
            limits.get("minimum_missing_reduction"),
            "promotion_limits.minimum_missing_reduction",
        ),
        maximum_spurious=_required_int(
            limits.get("maximum_spurious"),
            "promotion_limits.maximum_spurious",
        ),
        maximum_boundary=_required_int(
            limits.get("maximum_boundary"),
            "promotion_limits.maximum_boundary",
        ),
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _required_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _required_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


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
