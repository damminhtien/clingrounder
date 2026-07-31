"""Calibrate and apply a lightweight verifier over Phase 1 entity proposals."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.proposal_dataset import (
    Phase1ProposalDataset,
    Phase1ProposalExample,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import (
    PHASE1_PROPOSAL_FEATURE_CONTRACT,
    Phase1GenreBucket,
    ProposalSourceRole,
    extract_phase1_proposal_features,
    is_phase1_heading_only_proposal,
    phase1_genre_bucket,
)
from medical_kg_nlp.benchmarks.phase1.proposal_conflict_graph import (
    Phase1ConflictGraph,
    Phase1ConflictNode,
    build_phase1_conflict_graph,
    select_maximum_utility_nodes,
)
from medical_kg_nlp.evaluation.sparse_logistic import (
    SparseBinaryExample,
    SparseLogisticModel,
    SparseLogisticTrainingConfig,
    binary_probability_metrics,
    fit_sparse_logistic,
)
from medical_kg_nlp.ner.document_structure import DocumentStructureAnalyzer
from medical_kg_nlp.ontology.phase1 import (
    PHASE1_ALLOWED_TYPES,
    PHASE1_CODABLE_TYPES,
)
from medical_kg_nlp.utils.hashing import sha256_file

__all__ = [
    "Phase1ProbabilityCalibrator",
    "Phase1ProposalFitMode",
    "Phase1ProposalVerifier",
    "ScoredPhase1Proposal",
    "fit_phase1_proposal_verifier",
    "resolve_phase1_proposal_rows",
    "score_phase1_proposal_rows",
    "write_phase1_proposal_resolution",
    "write_phase1_proposal_verifier",
]

_VERIFIER_SCHEMA = "phase1-proposal-verifier.v4"
_CALIBRATOR_SCHEMA = "phase1-proposal-probability-calibrator.v1"
_DEFAULT_CROSS_FIT_FOLDS = 5
_MIN_GENRE_THRESHOLD_PROPOSALS = 20
_MIN_GENRE_THRESHOLD_POSITIVES = 3
_MIN_GENRE_THRESHOLD_NEGATIVES = 3
_CALIBRATOR_TRAINING_CONFIG = SparseLogisticTrainingConfig(
    epochs=500,
    learning_rate=0.15,
    learning_rate_decay=0.01,
    l2=0.0005,
    tolerance=1e-9,
)


class Phase1ProposalFitMode(StrEnum):
    """Choose between legacy split diagnostics and the governed final-fit model."""

    DEVELOPMENT = "development"
    FULL_OOF = "full_oof"


@dataclass(frozen=True, slots=True)
class Phase1ProbabilityCalibrator:
    """Selected probability mapping with an out-of-fold Platt candidate."""

    method: str
    model: SparseLogisticModel | None
    fold_count: int
    assignment_sha256: str

    def __post_init__(self) -> None:
        if self.method not in {"identity_logistic", "platt_document_grouped_oof"}:
            raise ValueError("Unsupported proposal probability calibration method")
        if (self.method == "identity_logistic") != (self.model is None):
            raise ValueError("Only Platt calibration may contain a calibrator model")
        if self.fold_count < 2:
            raise ValueError("Probability calibration requires at least two folds")
        if len(self.assignment_sha256) != 64:
            raise ValueError("Probability calibration requires a fold-assignment SHA-256")

    def predict_probability(
        self,
        base_logit: float,
        proposal_features: Mapping[str, float],
    ) -> float:
        """Map a base verifier logit to a calibrated exact-proposal probability."""

        if self.model is None:
            return _sigmoid(base_logit)
        return self.model.predict_probability(
            _probability_calibration_features(base_logit, proposal_features)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _CALIBRATOR_SCHEMA,
            "method": self.method,
            "fold_count": self.fold_count,
            "assignment_sha256": self.assignment_sha256,
            "model": self.model.to_dict() if self.model is not None else None,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> Phase1ProbabilityCalibrator:
        if payload.get("schema_version") != _CALIBRATOR_SCHEMA:
            raise ValueError("Unsupported proposal probability calibrator schema")
        method = str(payload.get("method", ""))
        raw_model = payload.get("model")
        if raw_model is not None and not isinstance(raw_model, Mapping):
            raise ValueError("Proposal probability calibrator model is invalid")
        return cls(
            method=method,
            model=(
                SparseLogisticModel.from_dict(raw_model)
                if isinstance(raw_model, Mapping)
                else None
            ),
            fold_count=int(payload.get("fold_count", 0)),
            assignment_sha256=str(payload.get("assignment_sha256", "")),
        )


@dataclass(frozen=True, slots=True)
class ScoredPhase1Proposal:
    """One runtime proposal and the verifier decision derived from it."""

    row: Mapping[str, Any]
    probability: float
    threshold: float
    selected_before_overlap: bool
    selected: bool
    rejection_reason: str | None
    genre: str = "unknown"
    conflict_component_id: str | None = None
    conflict_kinds: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": dict(self.row),
            "genre": self.genre,
            "probability": self.probability,
            "threshold": self.threshold,
            "selected_before_overlap": self.selected_before_overlap,
            "selected": self.selected,
            "rejection_reason": self.rejection_reason,
            "conflict_component_id": self.conflict_component_id,
            "conflict_kinds": list(self.conflict_kinds),
        }


@dataclass(frozen=True, slots=True)
class Phase1ProposalVerifier:
    """Portable verifier model with per-type thresholds calibrated on development."""

    model: SparseLogisticModel
    probability_calibrator: Phase1ProbabilityCalibrator
    thresholds: tuple[tuple[str, float], ...]
    training_dataset_sha256: str
    genre_probability_calibrators: tuple[
        tuple[str, Phase1ProbabilityCalibrator],
        ...,
    ] = ()
    genre_thresholds: tuple[tuple[str, str, float], ...] = ()
    minimum_development_precision: float | None = None
    fit_mode: Phase1ProposalFitMode = Phase1ProposalFitMode.DEVELOPMENT
    training_labels_scope: str = "legacy_train_development"

    def __post_init__(self) -> None:
        threshold_types = {entity_type for entity_type, _ in self.thresholds}
        if threshold_types != set(PHASE1_ALLOWED_TYPES):
            raise ValueError("Proposal verifier must define all Phase 1 type thresholds")
        if len(threshold_types) != len(self.thresholds):
            raise ValueError("Proposal verifier contains duplicate type thresholds")
        if any(not 0.0 <= threshold <= 1.0 for _, threshold in self.thresholds):
            raise ValueError("Proposal verifier thresholds must be within [0, 1]")
        if len(self.training_dataset_sha256) != 64:
            raise ValueError("Proposal verifier requires a dataset SHA-256")
        genre_names = [genre for genre, _ in self.genre_probability_calibrators]
        if len(genre_names) != len(set(genre_names)):
            raise ValueError("Proposal verifier contains duplicate genre calibrators")
        for genre in genre_names:
            Phase1GenreBucket(genre)
        genre_threshold_keys = [
            (genre, entity_type)
            for genre, entity_type, _ in self.genre_thresholds
        ]
        if len(genre_threshold_keys) != len(set(genre_threshold_keys)):
            raise ValueError("Proposal verifier contains duplicate genre thresholds")
        for genre, entity_type, threshold in self.genre_thresholds:
            Phase1GenreBucket(genre)
            if entity_type not in PHASE1_ALLOWED_TYPES:
                raise ValueError("Proposal verifier has an invalid genre threshold type")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("Proposal verifier genre thresholds must be within [0, 1]")
        if self.minimum_development_precision is not None and not (
            0.0 < self.minimum_development_precision <= 1.0
        ):
            raise ValueError("Minimum development precision must be within (0, 1]")
        if self.training_labels_scope not in {
            "legacy_train_development",
            "all_governed_manual_gold",
        }:
            raise ValueError("Proposal verifier training-label scope is invalid")
        if (
            self.fit_mode is Phase1ProposalFitMode.FULL_OOF
        ) != (self.training_labels_scope == "all_governed_manual_gold"):
            raise ValueError("Proposal verifier fit mode and label scope disagree")

    @property
    def threshold_by_type(self) -> dict[str, float]:
        return dict(self.thresholds)

    @property
    def genre_calibrator_by_bucket(self) -> dict[str, Phase1ProbabilityCalibrator]:
        """Return learned genre overrides; absent genres use the global calibrator."""

        return dict(self.genre_probability_calibrators)

    @property
    def genre_threshold_by_key(self) -> dict[tuple[str, str], float]:
        """Return learned genre/type operating points."""

        return {
            (genre, entity_type): threshold
            for genre, entity_type, threshold in self.genre_thresholds
        }

    def predict_probability(
        self,
        features: Mapping[str, float],
        *,
        genre: Phase1GenreBucket | str,
    ) -> float:
        """Return calibrated P(exact raw span and exact Phase 1 type)."""

        bucket = Phase1GenreBucket(genre)
        calibrator = self.genre_calibrator_by_bucket.get(
            bucket.value,
            self.probability_calibrator,
        )
        return calibrator.predict_probability(
            self.model.predict_logit(features),
            features,
        )

    def threshold_for(
        self,
        entity_type: str,
        *,
        genre: Phase1GenreBucket | str,
    ) -> float:
        """Return a genre/type threshold with an explicit global fallback."""

        bucket = Phase1GenreBucket(genre)
        return self.genre_threshold_by_key.get(
            (bucket.value, entity_type),
            self.threshold_by_type[entity_type],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _VERIFIER_SCHEMA,
            "feature_contract": PHASE1_PROPOSAL_FEATURE_CONTRACT,
            "training_dataset_sha256": self.training_dataset_sha256,
            "thresholds": dict(self.thresholds),
            "genre_thresholds": _nested_genre_thresholds(self.genre_thresholds),
            "minimum_development_precision": self.minimum_development_precision,
            "fit_mode": self.fit_mode.value,
            "training_labels_scope": self.training_labels_scope,
            "model": self.model.to_dict(),
            "probability_calibrator": self.probability_calibrator.to_dict(),
            "genre_probability_calibrators": {
                genre: calibrator.to_dict()
                for genre, calibrator in self.genre_probability_calibrators
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Phase1ProposalVerifier:
        if payload.get("schema_version") != _VERIFIER_SCHEMA:
            raise ValueError("Unsupported Phase 1 proposal verifier schema")
        if payload.get("feature_contract") != PHASE1_PROPOSAL_FEATURE_CONTRACT:
            raise ValueError("Proposal verifier feature contract is incompatible")
        raw_thresholds = payload.get("thresholds")
        raw_model = payload.get("model")
        raw_calibrator = payload.get("probability_calibrator")
        raw_genre_calibrators = payload.get("genre_probability_calibrators")
        raw_genre_thresholds = payload.get("genre_thresholds")
        if (
            not isinstance(raw_thresholds, Mapping)
            or not isinstance(raw_model, Mapping)
            or not isinstance(raw_calibrator, Mapping)
            or not isinstance(raw_genre_calibrators, Mapping)
            or not isinstance(raw_genre_thresholds, Mapping)
        ):
            raise ValueError("Proposal verifier artifact is incomplete")
        thresholds: list[tuple[str, float]] = []
        for entity_type, raw_threshold in raw_thresholds.items():
            if (
                not isinstance(entity_type, str)
                or not isinstance(raw_threshold, int | float)
                or isinstance(raw_threshold, bool)
            ):
                raise ValueError("Proposal verifier thresholds must be numeric")
            thresholds.append((entity_type, float(raw_threshold)))
        genre_calibrators: list[tuple[str, Phase1ProbabilityCalibrator]] = []
        for genre, raw_genre_calibrator in raw_genre_calibrators.items():
            if not isinstance(genre, str) or not isinstance(
                raw_genre_calibrator,
                Mapping,
            ):
                raise ValueError("Proposal verifier genre calibrators are invalid")
            genre_calibrators.append(
                (
                    genre,
                    Phase1ProbabilityCalibrator.from_dict(raw_genre_calibrator),
                )
            )
        genre_thresholds = _parse_genre_thresholds(raw_genre_thresholds)
        return cls(
            model=SparseLogisticModel.from_dict(raw_model),
            probability_calibrator=Phase1ProbabilityCalibrator.from_dict(
                raw_calibrator
            ),
            thresholds=tuple(sorted(thresholds)),
            training_dataset_sha256=str(payload.get("training_dataset_sha256", "")),
            genre_probability_calibrators=tuple(sorted(genre_calibrators)),
            genre_thresholds=genre_thresholds,
            minimum_development_precision=(
                float(payload["minimum_development_precision"])
                if isinstance(payload.get("minimum_development_precision"), int | float)
                and not isinstance(payload.get("minimum_development_precision"), bool)
                else None
            ),
            fit_mode=Phase1ProposalFitMode(str(payload.get("fit_mode", ""))),
            training_labels_scope=str(payload.get("training_labels_scope", "")),
        )


def fit_phase1_proposal_verifier(
    dataset: Phase1ProposalDataset,
    *,
    training_config: SparseLogisticTrainingConfig | None = None,
    minimum_development_precision: float | None = None,
    fit_mode: Phase1ProposalFitMode | str = Phase1ProposalFitMode.DEVELOPMENT,
) -> tuple[Phase1ProposalVerifier, dict[str, Any]]:
    """Fit a learned exact-span/type verifier and its probability operating point."""

    if dataset.manifest.get("feature_contract") != PHASE1_PROPOSAL_FEATURE_CONTRACT:
        raise ValueError("Proposal dataset feature contract is incompatible")
    if minimum_development_precision is not None and not (
        0.0 < minimum_development_precision <= 1.0
    ):
        raise ValueError("Minimum development precision must be within (0, 1]")
    active_fit_mode = Phase1ProposalFitMode(fit_mode)
    if active_fit_mode is Phase1ProposalFitMode.FULL_OOF:
        return _fit_phase1_proposal_verifier_full_oof(
            dataset,
            training_config=training_config,
            minimum_precision=minimum_development_precision,
        )
    train = tuple(example for example in dataset.examples if example.split == "train")
    development = tuple(
        example for example in dataset.examples if example.split == "development"
    )
    if not train or not development:
        raise ValueError("Proposal calibration requires train and development examples")

    active_training_config = training_config or SparseLogisticTrainingConfig()
    sparse_train = [
        SparseBinaryExample(
            features=example.features,
            label=example.label,
        )
        for example in train
    ]
    model, training_report = fit_sparse_logistic(
        sparse_train,
        config=active_training_config,
    )
    out_of_fold_logits, cross_fit_report = _cross_fitted_logits(
        train,
        config=active_training_config,
    )
    calibrator_examples = [
        SparseBinaryExample.from_mapping(
            _probability_calibration_features(
                logit,
                dict(example.features),
            ),
            label=example.label,
        )
        for example, logit in zip(train, out_of_fold_logits, strict=True)
    ]
    calibrator_model, calibrator_training_report = fit_sparse_logistic(
        calibrator_examples,
        config=_CALIBRATOR_TRAINING_CONFIG,
    )
    platt_calibrator = Phase1ProbabilityCalibrator(
        method="platt_document_grouped_oof",
        model=calibrator_model,
        fold_count=int(cross_fit_report["fold_count"]),
        assignment_sha256=str(cross_fit_report["assignment_sha256"]),
    )
    out_of_fold_raw_probabilities = tuple(
        _sigmoid(logit) for logit in out_of_fold_logits
    )
    out_of_fold_calibrated_probabilities = tuple(
        platt_calibrator.predict_probability(
            logit,
            dict(example.features),
        )
        for example, logit in zip(train, out_of_fold_logits, strict=True)
    )
    development_logits = tuple(
        model.predict_logit(dict(example.features)) for example in development
    )
    development_raw_probabilities = tuple(
        _sigmoid(logit) for logit in development_logits
    )
    development_platt_probabilities = tuple(
        platt_calibrator.predict_probability(
            model.predict_logit(dict(example.features)),
            dict(example.features),
        )
        for example in development
    )
    development_labels = [example.label for example in development]
    raw_development_metrics = binary_probability_metrics(
        development_labels,
        development_raw_probabilities,
    )
    platt_development_metrics = binary_probability_metrics(
        development_labels,
        development_platt_probabilities,
    )
    use_platt = _calibration_metric_rank(platt_development_metrics) < (
        _calibration_metric_rank(raw_development_metrics)
    )
    probability_calibrator = (
        platt_calibrator
        if use_platt
        else Phase1ProbabilityCalibrator(
            method="identity_logistic",
            model=None,
            fold_count=platt_calibrator.fold_count,
            assignment_sha256=platt_calibrator.assignment_sha256,
        )
    )
    genre_calibrators, genre_calibration_report = _fit_genre_calibrators(
        train,
        development,
        out_of_fold_logits,
        development_logits,
        global_calibrator=probability_calibrator,
        fold_count=platt_calibrator.fold_count,
        assignment_sha256=platt_calibrator.assignment_sha256,
    )
    development_probabilities = tuple(
        _predict_with_genre_calibrator(
            probability_calibrator,
            genre_calibrators,
            development_logits[index],
            dict(example.features),
            phase1_genre_bucket(example.genre),
        )
        for index, example in enumerate(development)
    )
    gold_counts = _gold_counts(dataset.manifest)
    gold_genre_counts = _gold_genre_counts(dataset.manifest)
    thresholds, threshold_report = _calibrate_thresholds(
        development,
        development_probabilities,
        gold_counts,
        minimum_precision=minimum_development_precision,
    )
    genre_thresholds, genre_threshold_report = _calibrate_genre_thresholds(
        development,
        development_probabilities,
        gold_genre_counts,
        minimum_precision=minimum_development_precision,
    )
    dataset_sha256 = _mapping_sha256(dataset.manifest)
    verifier = Phase1ProposalVerifier(
        model=model,
        probability_calibrator=probability_calibrator,
        thresholds=tuple(sorted(thresholds.items())),
        training_dataset_sha256=dataset_sha256,
        genre_probability_calibrators=tuple(sorted(genre_calibrators.items())),
        genre_thresholds=tuple(
            sorted(
                (
                    genre,
                    entity_type,
                    threshold,
                )
                for (genre, entity_type), threshold in genre_thresholds.items()
            )
        ),
        minimum_development_precision=minimum_development_precision,
        fit_mode=active_fit_mode,
        training_labels_scope="legacy_train_development",
    )
    learned_selected = _select_examples(
        development,
        development_probabilities,
        thresholds,
        genre_thresholds=genre_thresholds,
        resolve_overlaps=True,
    )
    report = {
        "schema_version": "phase1-proposal-calibration-report.v1",
        "feature_contract": PHASE1_PROPOSAL_FEATURE_CONTRACT,
        "training_dataset_sha256": dataset_sha256,
        "fit_mode": active_fit_mode.value,
        "holdout_opened": False,
        "decision_authority": "official_submission",
        "local_metrics_role": "diagnostic_only",
        "auto_promote": False,
        "operating_point": {
            "objective": (
                "maximum_recall_at_minimum_precision"
                if minimum_development_precision is not None
                else "maximum_f1"
            ),
            "minimum_development_precision": minimum_development_precision,
        },
        "training": {
            "base_model": training_report,
            "cross_fit": cross_fit_report,
            "probability_calibrator": calibrator_training_report,
        },
        "probability_metrics": {
            "train_out_of_fold_raw": binary_probability_metrics(
                [example.label for example in train],
                out_of_fold_raw_probabilities,
            ),
            "train_out_of_fold_calibrated": binary_probability_metrics(
                [example.label for example in train],
                out_of_fold_calibrated_probabilities,
            ),
            "development_raw": raw_development_metrics,
            "development_platt_oof": platt_development_metrics,
            "development_selected": binary_probability_metrics(
                development_labels,
                development_probabilities,
            ),
        },
        "threshold_calibration": threshold_report,
        "genre_threshold_calibration": genre_threshold_report,
        "development_selection": {
            "learned": _selection_metrics(
                learned_selected,
                gold_counts,
                split="development",
            ),
            **_development_baselines(development, gold_counts, dataset.manifest),
            "by_genre": _development_selection_by_genre(
                learned_selected,
                gold_genre_counts,
            ),
        },
        "coverage_ceiling": {
            split: _coverage_ceiling(dataset.examples, gold_counts, split=split)
            for split in ("train", "development")
        },
        "top_weights": _top_weights(model),
        "probability_calibration": {
            "selected_method": probability_calibrator.method,
            "selection_split": "development",
            "selection_objective": "minimum_brier_then_log_loss_then_ece",
            "fold_count": probability_calibrator.fold_count,
            "assignment_sha256": probability_calibrator.assignment_sha256,
            "candidates": {
                "identity_logistic": raw_development_metrics,
                "platt_document_grouped_oof": platt_development_metrics,
            },
            "platt_top_weights": _top_weights(calibrator_model, limit=10),
            "genre_overrides": genre_calibration_report,
        },
    }
    return verifier, report


def _fit_phase1_proposal_verifier_full_oof(
    dataset: Phase1ProposalDataset,
    *,
    training_config: SparseLogisticTrainingConfig | None,
    minimum_precision: float | None,
) -> tuple[Phase1ProposalVerifier, dict[str, Any]]:
    """Fit all supplied supervision while deriving probabilities from document-grouped OOF rows."""

    examples = tuple(dataset.examples)
    if not examples or {example.label for example in examples} != {0, 1}:
        raise ValueError("Full-OOF proposal calibration requires both proposal labels")
    active_training_config = training_config or SparseLogisticTrainingConfig()
    sparse_examples = [
        SparseBinaryExample(features=example.features, label=example.label)
        for example in examples
    ]
    model, training_report = fit_sparse_logistic(
        sparse_examples,
        config=active_training_config,
    )
    out_of_fold_logits, cross_fit_report = _cross_fitted_logits(
        examples,
        config=active_training_config,
    )
    calibrator_examples = [
        SparseBinaryExample.from_mapping(
            _probability_calibration_features(logit, dict(example.features)),
            label=example.label,
        )
        for example, logit in zip(examples, out_of_fold_logits, strict=True)
    ]
    calibrator_model, calibrator_training_report = fit_sparse_logistic(
        calibrator_examples,
        config=_CALIBRATOR_TRAINING_CONFIG,
    )
    platt_calibrator = Phase1ProbabilityCalibrator(
        method="platt_document_grouped_oof",
        model=calibrator_model,
        fold_count=int(cross_fit_report["fold_count"]),
        assignment_sha256=str(cross_fit_report["assignment_sha256"]),
    )
    raw_probabilities = tuple(_sigmoid(logit) for logit in out_of_fold_logits)
    nested_platt_probabilities, nested_calibration_report = (
        _cross_fitted_calibrator_probabilities(examples, out_of_fold_logits)
    )
    labels = [example.label for example in examples]
    raw_metrics = binary_probability_metrics(labels, raw_probabilities)
    platt_metrics = binary_probability_metrics(labels, nested_platt_probabilities)
    use_platt = _calibration_metric_rank(platt_metrics) < _calibration_metric_rank(
        raw_metrics
    )
    probability_calibrator = (
        platt_calibrator
        if use_platt
        else Phase1ProbabilityCalibrator(
            method="identity_logistic",
            model=None,
            fold_count=platt_calibrator.fold_count,
            assignment_sha256=platt_calibrator.assignment_sha256,
        )
    )
    operating_probabilities = (
        nested_platt_probabilities if use_platt else raw_probabilities
    )

    # MODEL: threshold fitting consumes only OOF predictions. Relabeling the in-memory diagnostic
    # view does not change source examples; it lets existing metric code aggregate all supplied
    # supervision without restoring the obsolete 60/16 promotion gate.
    operating_examples = tuple(
        replace(example, split="development") for example in examples
    )
    gold_counts = _combined_gold_counts(dataset.manifest)
    gold_genre_counts = _combined_gold_genre_counts(dataset.manifest)
    thresholds, threshold_report = _calibrate_thresholds(
        operating_examples,
        operating_probabilities,
        gold_counts,
        minimum_precision=minimum_precision,
    )
    genre_thresholds, genre_threshold_report = _calibrate_genre_thresholds(
        operating_examples,
        operating_probabilities,
        gold_genre_counts,
        minimum_precision=minimum_precision,
    )
    selected = _select_examples(
        operating_examples,
        operating_probabilities,
        thresholds,
        genre_thresholds=genre_thresholds,
        resolve_overlaps=True,
    )
    dataset_sha256 = _mapping_sha256(dataset.manifest)
    verifier = Phase1ProposalVerifier(
        model=model,
        probability_calibrator=probability_calibrator,
        thresholds=tuple(sorted(thresholds.items())),
        training_dataset_sha256=dataset_sha256,
        genre_thresholds=tuple(
            sorted(
                (genre, entity_type, threshold)
                for (genre, entity_type), threshold in genre_thresholds.items()
            )
        ),
        minimum_development_precision=minimum_precision,
        fit_mode=Phase1ProposalFitMode.FULL_OOF,
        training_labels_scope="all_governed_manual_gold",
    )
    report = {
        "schema_version": "phase1-proposal-calibration-report.v1",
        "feature_contract": PHASE1_PROPOSAL_FEATURE_CONTRACT,
        "training_dataset_sha256": dataset_sha256,
        "fit_mode": Phase1ProposalFitMode.FULL_OOF.value,
        "holdout_opened": bool(
            dataset.manifest.get("inputs", {}).get("holdout_labels_read", False)
            if isinstance(dataset.manifest.get("inputs"), Mapping)
            else False
        ),
        "decision_authority": "official_submission",
        "local_metrics_role": "diagnostic_only",
        "auto_promote": False,
        "operating_point": {
            "objective": (
                "maximum_recall_at_minimum_precision"
                if minimum_precision is not None
                else "maximum_f1"
            ),
            "minimum_development_precision": minimum_precision,
            "prediction_source": "document_grouped_out_of_fold",
        },
        "training": {
            "base_model": training_report,
            "cross_fit": cross_fit_report,
            "probability_calibrator": calibrator_training_report,
            "nested_probability_cross_fit": nested_calibration_report,
        },
        "probability_metrics": {
            "all_supervision_out_of_fold_raw": raw_metrics,
            "all_supervision_out_of_fold_platt": platt_metrics,
        },
        "probability_calibration": {
            "selected_method": probability_calibrator.method,
            "selection_split": "all_supervision_document_grouped_oof",
            "selection_objective": "minimum_brier_then_log_loss_then_ece",
            "fold_count": probability_calibrator.fold_count,
            "assignment_sha256": probability_calibrator.assignment_sha256,
            "candidates": {
                "identity_logistic": raw_metrics,
                "platt_document_grouped_oof": platt_metrics,
            },
            "platt_top_weights": _top_weights(calibrator_model, limit=10),
        },
        "threshold_calibration": threshold_report,
        "genre_threshold_calibration": genre_threshold_report,
        "diagnostic_selection": _selection_metrics(
            selected,
            gold_counts,
            split="development",
        ),
        "coverage_ceiling": _coverage_ceiling(
            operating_examples,
            gold_counts,
            split="development",
        ),
        "top_weights": _top_weights(model),
    }
    return verifier, report


def score_phase1_proposal_rows(
    rows: Sequence[Mapping[str, Any]],
    source_text_by_document: Mapping[str, str],
    verifier: Phase1ProposalVerifier,
    *,
    source_roles: Mapping[str, ProposalSourceRole | str],
) -> tuple[ScoredPhase1Proposal, ...]:
    """Score unlabeled proposals and apply calibrated thresholds plus overlap resolution."""

    analyzer = DocumentStructureAnalyzer()
    structures = {
        document_id: analyzer.analyze(source_text)
        for document_id, source_text in source_text_by_document.items()
    }
    preselected: list[tuple[Mapping[str, Any], float, float]] = []
    scored: list[ScoredPhase1Proposal] = []
    for row in rows:
        document_id = str(row.get("document_id", ""))
        if document_id not in source_text_by_document:
            raise ValueError(f"Proposal references unknown document {document_id!r}")
        entity_type = str(row.get("type", ""))
        if entity_type not in PHASE1_ALLOWED_TYPES:
            raise ValueError(f"Proposal has unsupported type {entity_type!r}")
        features = extract_phase1_proposal_features(
            row,
            source_text_by_document[document_id],
            source_roles,
            structure=structures[document_id],
        )
        genre = phase1_genre_bucket(structures[document_id].genre)
        probability = verifier.predict_probability(features, genre=genre)
        threshold = verifier.threshold_for(entity_type, genre=genre)
        structural_heading = is_phase1_heading_only_proposal(
            row,
            source_text_by_document[document_id],
            structure=structures[document_id],
        )
        selected_before_overlap = probability >= threshold and not structural_heading
        if selected_before_overlap:
            preselected.append((row, probability, threshold))
        scored.append(
            ScoredPhase1Proposal(
                row=row,
                genre=genre.value,
                probability=probability,
                threshold=threshold,
                selected_before_overlap=selected_before_overlap,
                selected=False,
                rejection_reason=(
                    "structural_heading"
                    if structural_heading
                    else (
                        "below_threshold"
                        if probability < threshold
                        else "overlap_pending"
                    )
                ),
            )
        )

    accepted_ids, conflict_trace = _resolve_runtime_overlaps(preselected)
    return tuple(
        ScoredPhase1Proposal(
            row=item.row,
            genre=item.genre,
            probability=item.probability,
            threshold=item.threshold,
            selected_before_overlap=item.selected_before_overlap,
            selected=id(item.row) in accepted_ids,
            rejection_reason=None
            if id(item.row) in accepted_ids
            else (
                "overlap"
                if item.selected_before_overlap
                else item.rejection_reason
            ),
            conflict_component_id=conflict_trace.get(id(item.row), (None, ()))[0],
            conflict_kinds=conflict_trace.get(id(item.row), (None, ()))[1],
        )
        for item in scored
    )


def resolve_phase1_proposal_rows(
    rows: Sequence[Mapping[str, Any]],
    source_text_by_document: Mapping[str, str],
    verifier: Phase1ProposalVerifier,
    *,
    source_roles: Mapping[str, ProposalSourceRole | str],
) -> tuple[dict[str, list[dict[str, Any]]], tuple[ScoredPhase1Proposal, ...]]:
    """Resolve a proposal union into non-overlapping entity-only Phase 1 rows."""

    scored = score_phase1_proposal_rows(
        rows,
        source_text_by_document,
        verifier,
        source_roles=source_roles,
    )
    output: dict[str, list[dict[str, Any]]] = {
        document_id: [] for document_id in source_text_by_document
    }
    for item in scored:
        if not item.selected:
            continue
        row = item.row
        document_id = str(row.get("document_id", ""))
        position = _row_position(row)
        source_text = source_text_by_document[document_id]
        if source_text[position[0] : position[1]] != row.get("text"):
            raise ValueError("Selected proposal no longer matches its raw source offset")
        entity_type = str(row.get("type", ""))
        entity = {
            "text": str(row.get("text", "")),
            "type": entity_type,
            "assertions": [],
            "position": [position[0], position[1]],
        }
        if entity_type in PHASE1_CODABLE_TYPES:
            entity["candidates"] = []
        output[document_id].append(entity)
    for document_rows in output.values():
        document_rows.sort(key=_phase1_output_sort_key)
    return output, scored


def write_phase1_proposal_verifier(
    verifier: Phase1ProposalVerifier,
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    """Write a portable model and an inspectable calibration report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "verifier.json").write_text(
        json.dumps(verifier.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_phase1_proposal_resolution(
    rows_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    scored: Sequence[ScoredPhase1Proposal],
    output_dir: str | Path,
    *,
    matrix_path: str | Path,
    verifier_path: str | Path,
    source_roles: Mapping[str, ProposalSourceRole | str],
) -> dict[str, Any]:
    """Write deterministic resolved rows, full decision trace, and provenance."""

    output = Path(output_dir)
    entities_dir = output / "output"
    entities_dir.mkdir(parents=True, exist_ok=True)
    for document_id in sorted(rows_by_document, key=_document_sort_key):
        (entities_dir / f"{document_id}.json").write_text(
            json.dumps(
                list(rows_by_document[document_id]),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    (output / "scores.jsonl").write_text(
        "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in scored
        ),
        encoding="utf-8",
    )
    rejection_counts = Counter(
        item.rejection_reason or "selected"
        for item in scored
    )
    verifier_payload = json.loads(Path(verifier_path).read_text(encoding="utf-8"))
    if not isinstance(verifier_payload, Mapping):
        raise ValueError("Proposal verifier artifact must be an object")
    training_labels_scope = str(
        verifier_payload.get("training_labels_scope", "")
    )
    manifest = {
        "schema_version": "phase1-proposal-resolution.v1",
        "feature_contract": PHASE1_PROPOSAL_FEATURE_CONTRACT,
        "holdout_labels_opened": (
            training_labels_scope == "all_governed_manual_gold"
        ),
        "verifier_fit_mode": verifier_payload.get("fit_mode"),
        "training_labels_scope": training_labels_scope,
        "inputs": {
            "matrix": {
                "path": str(matrix_path),
                "sha256": sha256_file(matrix_path),
            },
            "verifier": {
                "path": str(verifier_path),
                "sha256": sha256_file(verifier_path),
            },
        },
        "source_roles": {
            source: ProposalSourceRole(role).value
            for source, role in sorted(source_roles.items())
        },
        "document_count": len(rows_by_document),
        "proposal_count": len(scored),
        "selected_count": sum(item.selected for item in scored),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _cross_fitted_logits(
    examples: Sequence[Phase1ProposalExample],
    *,
    config: SparseLogisticTrainingConfig,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Generate train logits without scoring a document with a model that saw it."""

    document_ids = sorted(
        {example.document_id for example in examples},
        key=_document_sort_key,
    )
    fold_by_document = _select_grouped_folds(examples, document_ids)
    logits: list[float | None] = [None] * len(examples)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(max(fold_by_document.values()) + 1):
        training_indices = [
            index
            for index, example in enumerate(examples)
            if fold_by_document[example.document_id] != fold
        ]
        validation_indices = [
            index
            for index, example in enumerate(examples)
            if fold_by_document[example.document_id] == fold
        ]
        fold_model, _ = fit_sparse_logistic(
            [
                SparseBinaryExample(
                    features=examples[index].features,
                    label=examples[index].label,
                )
                for index in training_indices
            ],
            config=config,
        )
        for index in validation_indices:
            logits[index] = fold_model.predict_logit(dict(examples[index].features))
        fold_reports.append(
            {
                "fold": fold,
                "train_example_count": len(training_indices),
                "validation_example_count": len(validation_indices),
                "validation_document_count": len(
                    {
                        examples[index].document_id
                        for index in validation_indices
                    }
                ),
            }
        )
    if any(value is None for value in logits):
        raise RuntimeError("Cross-fitting did not score every proposal example")
    assignment = "\n".join(
        f"{document_id}:{fold_by_document[document_id]}"
        for document_id in document_ids
    )
    return (
        tuple(float(value) for value in logits if value is not None),
        {
            "method": "document_grouped_round_robin",
            "fold_count": max(fold_by_document.values()) + 1,
            "document_count": len(document_ids),
            "assignment_sha256": hashlib.sha256(
                assignment.encode("utf-8")
            ).hexdigest(),
            "folds": fold_reports,
        },
    )


def _cross_fitted_calibrator_probabilities(
    examples: Sequence[Phase1ProposalExample],
    base_logits: Sequence[float],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Evaluate Platt calibration without scoring a document used to fit that mapping."""

    document_ids = sorted(
        {example.document_id for example in examples},
        key=_document_sort_key,
    )
    fold_by_document = _select_grouped_folds(examples, document_ids)
    probabilities: list[float | None] = [None] * len(examples)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(max(fold_by_document.values()) + 1):
        training_indices = [
            index
            for index, example in enumerate(examples)
            if fold_by_document[example.document_id] != fold
        ]
        validation_indices = [
            index
            for index, example in enumerate(examples)
            if fold_by_document[example.document_id] == fold
        ]
        calibrator_model, _ = fit_sparse_logistic(
            [
                SparseBinaryExample.from_mapping(
                    _probability_calibration_features(
                        base_logits[index],
                        dict(examples[index].features),
                    ),
                    label=examples[index].label,
                )
                for index in training_indices
            ],
            config=_CALIBRATOR_TRAINING_CONFIG,
        )
        for index in validation_indices:
            probabilities[index] = calibrator_model.predict_probability(
                _probability_calibration_features(
                    base_logits[index],
                    dict(examples[index].features),
                )
            )
        fold_reports.append(
            {
                "fold": fold,
                "train_example_count": len(training_indices),
                "validation_example_count": len(validation_indices),
                "validation_document_count": len(
                    {
                        examples[index].document_id
                        for index in validation_indices
                    }
                ),
            }
        )
    if any(value is None for value in probabilities):
        raise RuntimeError("Calibrator cross-fitting did not score every proposal")
    return (
        tuple(float(value) for value in probabilities if value is not None),
        {
            "method": "document_grouped_nested_platt",
            "fold_count": max(fold_by_document.values()) + 1,
            "document_count": len(document_ids),
            "folds": fold_reports,
        },
    )


def _fit_genre_calibrators(
    train: Sequence[Phase1ProposalExample],
    development: Sequence[Phase1ProposalExample],
    out_of_fold_logits: Sequence[float],
    development_logits: Sequence[float],
    *,
    global_calibrator: Phase1ProbabilityCalibrator,
    fold_count: int,
    assignment_sha256: str,
) -> tuple[dict[str, Phase1ProbabilityCalibrator], dict[str, Any]]:
    """Fit genre-specific probability maps only when labeled support proves useful."""

    selected: dict[str, Phase1ProbabilityCalibrator] = {}
    report: dict[str, Any] = {}
    for bucket in (
        Phase1GenreBucket.CLINICAL,
        Phase1GenreBucket.QUESTION_ANSWER,
        Phase1GenreBucket.EDUCATIONAL,
    ):
        train_indices = [
            index
            for index, example in enumerate(train)
            if phase1_genre_bucket(example.genre) is bucket
        ]
        development_indices = [
            index
            for index, example in enumerate(development)
            if phase1_genre_bucket(example.genre) is bucket
        ]
        positive_count = sum(train[index].label for index in train_indices)
        negative_count = len(train_indices) - positive_count
        genre_report: dict[str, Any] = {
            "train_support": len(train_indices),
            "train_positive": positive_count,
            "train_negative": negative_count,
            "development_support": len(development_indices),
            "selected": False,
        }
        if (
            len(train_indices) < _MIN_GENRE_THRESHOLD_PROPOSALS
            or positive_count < _MIN_GENRE_THRESHOLD_POSITIVES
            or negative_count < _MIN_GENRE_THRESHOLD_NEGATIVES
        ):
            genre_report["fallback_reason"] = "insufficient_train_support"
            report[bucket.value] = genre_report
            continue
        if not development_indices:
            genre_report["fallback_reason"] = "no_development_support"
            report[bucket.value] = genre_report
            continue

        calibrator_model, training_report = fit_sparse_logistic(
            [
                SparseBinaryExample.from_mapping(
                    _probability_calibration_features(
                        out_of_fold_logits[index],
                        dict(train[index].features),
                    ),
                    label=train[index].label,
                )
                for index in train_indices
            ],
            config=SparseLogisticTrainingConfig(
                epochs=500,
                learning_rate=0.15,
                learning_rate_decay=0.01,
                l2=0.0005,
                tolerance=1e-9,
            ),
        )
        candidate = Phase1ProbabilityCalibrator(
            method="platt_document_grouped_oof",
            model=calibrator_model,
            fold_count=fold_count,
            assignment_sha256=assignment_sha256,
        )
        labels = [development[index].label for index in development_indices]
        global_probabilities = [
            global_calibrator.predict_probability(
                development_logits[index],
                dict(development[index].features),
            )
            for index in development_indices
        ]
        candidate_probabilities = [
            candidate.predict_probability(
                development_logits[index],
                dict(development[index].features),
            )
            for index in development_indices
        ]
        global_metrics = binary_probability_metrics(labels, global_probabilities)
        candidate_metrics = binary_probability_metrics(labels, candidate_probabilities)
        use_candidate = _calibration_metric_rank(candidate_metrics) < (
            _calibration_metric_rank(global_metrics)
        )
        if use_candidate:
            selected[bucket.value] = candidate
        genre_report.update(
            {
                "selected": use_candidate,
                "fallback_reason": None if use_candidate else "global_better_or_equal",
                "selection_objective": "minimum_brier_then_log_loss_then_ece",
                "global_metrics": global_metrics,
                "genre_candidate_metrics": candidate_metrics,
                "training": training_report,
            }
        )
        report[bucket.value] = genre_report
    return selected, report


def _predict_with_genre_calibrator(
    global_calibrator: Phase1ProbabilityCalibrator,
    genre_calibrators: Mapping[str, Phase1ProbabilityCalibrator],
    base_logit: float,
    features: Mapping[str, float],
    genre: Phase1GenreBucket,
) -> float:
    calibrator = genre_calibrators.get(genre.value, global_calibrator)
    return calibrator.predict_probability(base_logit, features)


def _select_grouped_folds(
    examples: Sequence[Phase1ProposalExample],
    document_ids: Sequence[str],
) -> dict[str, int]:
    """Choose the largest deterministic fold count with valid binary train folds."""

    maximum = min(_DEFAULT_CROSS_FIT_FOLDS, len(document_ids))
    for fold_count in range(maximum, 1, -1):
        assignment = {
            document_id: index % fold_count
            for index, document_id in enumerate(document_ids)
        }
        valid = True
        for fold in range(fold_count):
            labels = {
                example.label
                for example in examples
                if assignment[example.document_id] != fold
            }
            if labels != {0, 1}:
                valid = False
                break
        if valid:
            return assignment
    raise ValueError(
        "Proposal probability calibration needs at least two document groups "
        "whose complementary train folds contain both labels"
    )


def _probability_calibration_features(
    base_logit: float,
    proposal_features: Mapping[str, float],
) -> dict[str, float]:
    """Build the small Platt feature vector, including per-type base-rate shifts."""

    features = {"base_logit": float(base_logit)}
    for name, value in proposal_features.items():
        if name.startswith("type:"):
            features[f"calibration:{name}"] = float(value)
    return features


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(max(value, -60.0))
    return exponent / (1.0 + exponent)


def _calibration_metric_rank(
    metrics: Mapping[str, float | int],
) -> tuple[float, float, float]:
    """Rank calibration mappings without using the downstream F1 threshold."""

    return (
        float(metrics["brier"]),
        float(metrics["log_loss"]),
        float(metrics["expected_calibration_error"]),
    )


def _calibrate_thresholds(
    examples: Sequence[Phase1ProposalExample],
    probabilities: Sequence[float],
    gold_counts: Mapping[tuple[str, str], int],
    *,
    minimum_precision: float | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds: dict[str, float] = {}
    report: dict[str, Any] = {}
    for entity_type in sorted(PHASE1_ALLOWED_TYPES):
        indices = [
            index
            for index, example in enumerate(examples)
            if example.entity_type == entity_type
        ]
        type_examples = tuple(examples[index] for index in indices)
        type_probabilities = tuple(probabilities[index] for index in indices)
        candidates = sorted({0.0, 1.0, *type_probabilities})
        trials: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
        for threshold in candidates:
            selected = _select_examples(
                type_examples,
                type_probabilities,
                {entity_type: threshold},
                resolve_overlaps=True,
            )
            metrics = _selection_metrics(
                selected,
                gold_counts,
                split="development",
                entity_type=entity_type,
            )
            if minimum_precision is None:
                # Precision and a higher threshold break equal-F1 ties conservatively.
                rank = (
                    float(metrics["f1"]),
                    float(metrics["precision"]),
                    float(metrics["recall"]),
                    threshold,
                )
            else:
                if (
                    int(metrics["true_positive"]) == 0
                    or float(metrics["precision"]) < minimum_precision
                ):
                    continue
                rank = (
                    float(metrics["recall"]),
                    float(metrics["f1"]),
                    float(metrics["precision"]),
                    threshold,
                )
            trials.append((rank, threshold, metrics))
        if trials:
            _, best_threshold, best_metrics = max(trials, key=lambda item: item[0])
        else:
            best_threshold = 1.0
            best_metrics = _selection_metrics(
                (),
                gold_counts,
                split="development",
                entity_type=entity_type,
            )
        thresholds[entity_type] = best_threshold
        report[entity_type] = {
            "proposal_support": len(type_examples),
            "positive_proposal_support": sum(
                example.label for example in type_examples
            ),
            "gold_support": gold_counts.get(("development", entity_type), 0),
            "candidate_threshold_count": len(candidates),
            "minimum_precision": minimum_precision,
            "threshold": best_threshold,
            "metrics": best_metrics,
        }
    return thresholds, report


def _calibrate_genre_thresholds(
    examples: Sequence[Phase1ProposalExample],
    probabilities: Sequence[float],
    gold_counts: Mapping[tuple[str, str, str], int],
    *,
    minimum_precision: float | None,
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Select genre/type overrides only where development support is defensible."""

    thresholds: dict[tuple[str, str], float] = {}
    report: dict[str, Any] = {}
    for bucket in (
        Phase1GenreBucket.CLINICAL,
        Phase1GenreBucket.QUESTION_ANSWER,
        Phase1GenreBucket.EDUCATIONAL,
    ):
        bucket_report: dict[str, Any] = {}
        for entity_type in sorted(PHASE1_ALLOWED_TYPES):
            indices = [
                index
                for index, example in enumerate(examples)
                if phase1_genre_bucket(example.genre) is bucket
                and example.entity_type == entity_type
            ]
            type_examples = tuple(examples[index] for index in indices)
            type_probabilities = tuple(probabilities[index] for index in indices)
            positive_count = sum(example.label for example in type_examples)
            negative_count = len(type_examples) - positive_count
            gold = gold_counts.get(
                ("development", bucket.value, entity_type),
                0,
            )
            type_report: dict[str, Any] = {
                "proposal_support": len(type_examples),
                "positive_proposal_support": positive_count,
                "negative_proposal_support": negative_count,
                "gold_support": gold,
                "selected": False,
            }
            if (
                len(type_examples) < _MIN_GENRE_THRESHOLD_PROPOSALS
                or positive_count < _MIN_GENRE_THRESHOLD_POSITIVES
                or negative_count < _MIN_GENRE_THRESHOLD_NEGATIVES
                or gold < _MIN_GENRE_THRESHOLD_POSITIVES
            ):
                type_report["fallback_reason"] = "insufficient_development_support"
                bucket_report[entity_type] = type_report
                continue

            candidates = sorted({0.0, 1.0, *type_probabilities})
            trials: list[
                tuple[tuple[float, float, float, float], float, dict[str, Any]]
            ] = []
            for threshold in candidates:
                selected = _select_examples(
                    type_examples,
                    type_probabilities,
                    {entity_type: threshold},
                    resolve_overlaps=True,
                )
                metrics = _metrics_against_gold(selected, gold)
                if minimum_precision is None:
                    rank = (
                        float(metrics["f1"]),
                        float(metrics["precision"]),
                        float(metrics["recall"]),
                        threshold,
                    )
                else:
                    if (
                        int(metrics["true_positive"]) == 0
                        or float(metrics["precision"]) < minimum_precision
                    ):
                        continue
                    rank = (
                        float(metrics["recall"]),
                        float(metrics["f1"]),
                        float(metrics["precision"]),
                        threshold,
                    )
                trials.append((rank, threshold, metrics))
            if not trials:
                type_report["fallback_reason"] = "no_threshold_met_precision_gate"
                bucket_report[entity_type] = type_report
                continue
            _, threshold, metrics = max(trials, key=lambda item: item[0])
            thresholds[(bucket.value, entity_type)] = threshold
            type_report.update(
                {
                    "selected": True,
                    "fallback_reason": None,
                    "threshold": threshold,
                    "candidate_threshold_count": len(candidates),
                    "metrics": metrics,
                }
            )
            bucket_report[entity_type] = type_report
        report[bucket.value] = bucket_report
    return thresholds, report


def _select_examples(
    examples: Sequence[Phase1ProposalExample],
    probabilities: Sequence[float],
    thresholds: Mapping[str, float],
    *,
    genre_thresholds: Mapping[tuple[str, str], float] | None = None,
    resolve_overlaps: bool,
) -> tuple[Phase1ProposalExample, ...]:
    active_genre_thresholds = genre_thresholds or {}
    candidates = [
        (example, probability)
        for example, probability in zip(examples, probabilities, strict=True)
        if probability
        >= active_genre_thresholds.get(
            (
                phase1_genre_bucket(example.genre).value,
                example.entity_type,
            ),
            thresholds[example.entity_type],
        )
    ]
    if not resolve_overlaps:
        return tuple(example for example, _ in candidates)
    example_by_node_id: dict[str, Phase1ProposalExample] = {}
    nodes: list[Phase1ConflictNode] = []
    for index, (example, probability) in enumerate(candidates):
        node_id = f"diagnostic:{index:08d}:{example.proposal_id}"
        example_by_node_id[node_id] = example
        threshold = active_genre_thresholds.get(
            (
                phase1_genre_bucket(example.genre).value,
                example.entity_type,
            ),
            thresholds[example.entity_type],
        )
        nodes.append(
            Phase1ConflictNode(
                node_id=node_id,
                document_id=example.document_id,
                span=example.position,
                entity_type=example.entity_type,
                probability=probability,
                source_count=max(1, len(example.sources)),
                decision_threshold=threshold,
            )
        )
    graph = build_phase1_conflict_graph(tuple(nodes))
    return tuple(
        example_by_node_id[node.node_id]
        for node in select_maximum_utility_nodes(graph)
    )


def _resolve_runtime_overlaps(
    rows: Sequence[tuple[Mapping[str, Any], float, float]],
) -> tuple[set[int], dict[int, tuple[str | None, tuple[str, ...]]]]:
    nodes: list[Phase1ConflictNode] = []
    row_id_by_node_id: dict[str, int] = {}
    for index, (row, probability, threshold) in enumerate(rows):
        raw_sources = row.get("sources")
        source_count = (
            len(raw_sources)
            if isinstance(raw_sources, list) and raw_sources
            else int(row.get("source_count", 1))
        )
        node_id = f"runtime:{index:08d}"
        row_id_by_node_id[node_id] = id(row)
        nodes.append(
            Phase1ConflictNode(
                node_id=node_id,
                document_id=str(row.get("document_id", "")),
                span=_row_position(row),
                entity_type=str(row.get("type", "")),
                probability=probability,
                source_count=source_count,
                decision_threshold=threshold,
            )
        )
    graph = build_phase1_conflict_graph(tuple(nodes))
    accepted_ids = {
        row_id_by_node_id[node.node_id]
        for node in select_maximum_utility_nodes(graph)
    }
    return accepted_ids, _runtime_conflict_trace(graph, row_id_by_node_id)


def _runtime_conflict_trace(
    graph: Phase1ConflictGraph,
    row_id_by_node_id: Mapping[str, int],
) -> dict[int, tuple[str | None, tuple[str, ...]]]:
    trace: dict[int, tuple[str | None, tuple[str, ...]]] = {}
    for component in graph.components:
        kinds_by_node: dict[str, set[str]] = {
            node_id: set() for node_id in component.node_ids
        }
        for edge in component.edges:
            kinds_by_node[edge.left_id].add(edge.kind.value)
            kinds_by_node[edge.right_id].add(edge.kind.value)
        component_id = component.component_id if component.edges else None
        for node_id, kinds in kinds_by_node.items():
            trace[row_id_by_node_id[node_id]] = (
                component_id,
                tuple(sorted(kinds)),
            )
    return trace


def _selection_metrics(
    selected: Sequence[Phase1ProposalExample],
    gold_counts: Mapping[tuple[str, str], int],
    *,
    split: str,
    entity_type: str | None = None,
) -> dict[str, Any]:
    scoped = [
        example
        for example in selected
        if example.split == split
        and (entity_type is None or example.entity_type == entity_type)
    ]
    gold = sum(
        count
        for (count_split, count_type), count in gold_counts.items()
        if count_split == split and (entity_type is None or count_type == entity_type)
    )
    return _metrics_against_gold(scoped, gold)


def _metrics_against_gold(
    selected: Sequence[Phase1ProposalExample],
    gold: int,
) -> dict[str, Any]:
    true_positive = sum(example.label for example in selected)
    false_positive = len(selected) - true_positive
    false_negative = max(0, gold - true_positive)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / gold if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted": len(selected),
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "selected_error_counts": dict(
            sorted(Counter(example.error_kind for example in selected).items())
        ),
    }


def _development_selection_by_genre(
    selected: Sequence[Phase1ProposalExample],
    gold_counts: Mapping[tuple[str, str, str], int],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for bucket in Phase1GenreBucket:
        scoped = [
            example
            for example in selected
            if example.split == "development"
            and phase1_genre_bucket(example.genre) is bucket
        ]
        gold = sum(
            count
            for (split, genre, _), count in gold_counts.items()
            if split == "development" and genre == bucket.value
        )
        if scoped or gold:
            report[bucket.value] = _metrics_against_gold(scoped, gold)
    return report


def _development_baselines(
    examples: Sequence[Phase1ProposalExample],
    gold_counts: Mapping[tuple[str, str], int],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    baselines: dict[str, Any] = {
        "union": _selection_metrics(
            examples,
            gold_counts,
            split="development",
        ),
        "exact_agreement": _selection_metrics(
            [example for example in examples if example.status == "exact_agreement"],
            gold_counts,
            split="development",
        ),
    }
    raw_roles = manifest.get("source_roles")
    if isinstance(raw_roles, Mapping):
        for source, role in sorted(raw_roles.items()):
            selected = [example for example in examples if source in example.sources]
            baselines[f"source:{source}:{role}"] = _selection_metrics(
                selected,
                gold_counts,
                split="development",
            )
    return baselines


def _coverage_ceiling(
    examples: Sequence[Phase1ProposalExample],
    gold_counts: Mapping[tuple[str, str], int],
    *,
    split: str,
) -> dict[str, Any]:
    positive = sum(
        example.label for example in examples if example.split == split
    )
    gold = sum(
        count for (count_split, _), count in gold_counts.items() if count_split == split
    )
    return {
        "covered_gold": positive,
        "gold": gold,
        "recall": positive / gold if gold else 0.0,
    }


def _gold_counts(manifest: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    raw = manifest.get("gold_entity_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("Proposal dataset manifest has no gold entity counts")
    counts: dict[tuple[str, str], int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise ValueError("Proposal gold entity counts are malformed")
        split, separator, entity_type = key.partition(":")
        if not separator or split not in {"train", "development"}:
            raise ValueError("Proposal gold count key is malformed")
        counts[(split, entity_type)] = value
    return counts


def _combined_gold_counts(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, str], int]:
    """Aggregate legacy splits into the diagnostic view used by full-OOF fitting."""

    combined: Counter[str] = Counter()
    for (_, entity_type), count in _gold_counts(manifest).items():
        combined[entity_type] += count
    return {
        ("development", entity_type): combined.get(entity_type, 0)
        for entity_type in PHASE1_ALLOWED_TYPES
    }


def _gold_genre_counts(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, str, str], int]:
    raw = manifest.get("gold_entity_genre_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("Proposal dataset manifest has no genre-specific gold counts")
    counts: dict[tuple[str, str, str], int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise ValueError("Proposal genre-specific gold counts are malformed")
        split, separator, remainder = key.partition(":")
        genre, second_separator, entity_type = remainder.partition(":")
        if (
            not separator
            or not second_separator
            or split not in {"train", "development"}
            or entity_type not in PHASE1_ALLOWED_TYPES
        ):
            raise ValueError("Proposal genre-specific gold count key is malformed")
        Phase1GenreBucket(genre)
        counts[(split, genre, entity_type)] = value
    return counts


def _combined_gold_genre_counts(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, str, str], int]:
    """Aggregate legacy split labels while preserving genre/type support."""

    combined: Counter[tuple[str, str]] = Counter()
    for (_, genre, entity_type), count in _gold_genre_counts(manifest).items():
        combined[(genre, entity_type)] += count
    return {
        ("development", genre.value, entity_type): combined.get(
            (genre.value, entity_type),
            0,
        )
        for genre in Phase1GenreBucket
        for entity_type in PHASE1_ALLOWED_TYPES
    }


def _nested_genre_thresholds(
    thresholds: Sequence[tuple[str, str, float]],
) -> dict[str, dict[str, float]]:
    nested: dict[str, dict[str, float]] = {}
    for genre, entity_type, threshold in thresholds:
        nested.setdefault(genre, {})[entity_type] = threshold
    return {
        genre: dict(sorted(values.items()))
        for genre, values in sorted(nested.items())
    }


def _parse_genre_thresholds(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, str, float], ...]:
    thresholds: list[tuple[str, str, float]] = []
    for genre, raw_values in payload.items():
        if not isinstance(genre, str) or not isinstance(raw_values, Mapping):
            raise ValueError("Proposal verifier genre thresholds are invalid")
        Phase1GenreBucket(genre)
        for entity_type, raw_threshold in raw_values.items():
            if (
                not isinstance(entity_type, str)
                or not isinstance(raw_threshold, int | float)
                or isinstance(raw_threshold, bool)
            ):
                raise ValueError("Proposal verifier genre thresholds must be numeric")
            thresholds.append((genre, entity_type, float(raw_threshold)))
    return tuple(sorted(thresholds))


def _top_weights(model: SparseLogisticModel, limit: int = 25) -> dict[str, Any]:
    pairs = list(zip(model.feature_names, model.weights, strict=True))
    return {
        "positive": [
            {"feature": name, "weight": weight}
            for name, weight in sorted(pairs, key=lambda pair: (-pair[1], pair[0]))[:limit]
        ],
        "negative": [
            {"feature": name, "weight": weight}
            for name, weight in sorted(pairs, key=lambda pair: (pair[1], pair[0]))[:limit]
        ],
    }




def _row_position(row: Mapping[str, Any]) -> tuple[int, int]:
    value = row.get("position")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ValueError("Proposal has an invalid position")
    return value[0], value[1]


def _phase1_output_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    start, end = _row_position(row)
    return start, end, str(row.get("type", ""))


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
