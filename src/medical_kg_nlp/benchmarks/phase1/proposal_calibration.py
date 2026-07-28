"""Calibrate and apply a lightweight verifier over Phase 1 entity proposals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.proposal_dataset import (
    Phase1ProposalDataset,
    Phase1ProposalExample,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import (
    PHASE1_PROPOSAL_FEATURE_CONTRACT,
    ProposalSourceRole,
    extract_phase1_proposal_features,
    is_phase1_heading_only_proposal,
)
from medical_kg_nlp.evaluation.sparse_logistic import (
    SparseBinaryExample,
    SparseLogisticModel,
    SparseLogisticTrainingConfig,
    binary_probability_metrics,
    fit_sparse_logistic,
)
from medical_kg_nlp.ner.document_structure import DocumentStructureAnalyzer
from medical_kg_nlp.ontology.phase1 import PHASE1_ALLOWED_TYPES, PHASE1_TYPE_PRIORITY

__all__ = [
    "Phase1ProposalVerifier",
    "ScoredPhase1Proposal",
    "fit_phase1_proposal_verifier",
    "score_phase1_proposal_rows",
    "write_phase1_proposal_verifier",
]

_VERIFIER_SCHEMA = "phase1-proposal-verifier.v1"


@dataclass(frozen=True, slots=True)
class ScoredPhase1Proposal:
    """One runtime proposal and the verifier decision derived from it."""

    row: Mapping[str, Any]
    probability: float
    threshold: float
    selected_before_overlap: bool
    selected: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": dict(self.row),
            "probability": self.probability,
            "threshold": self.threshold,
            "selected_before_overlap": self.selected_before_overlap,
            "selected": self.selected,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True, slots=True)
class Phase1ProposalVerifier:
    """Portable verifier model with per-type thresholds calibrated on development."""

    model: SparseLogisticModel
    thresholds: tuple[tuple[str, float], ...]
    training_dataset_sha256: str
    minimum_development_precision: float | None = None

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
        if self.minimum_development_precision is not None and not (
            0.0 < self.minimum_development_precision <= 1.0
        ):
            raise ValueError("Minimum development precision must be within (0, 1]")

    @property
    def threshold_by_type(self) -> dict[str, float]:
        return dict(self.thresholds)

    def predict_probability(self, features: Mapping[str, float]) -> float:
        return self.model.predict_probability(features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _VERIFIER_SCHEMA,
            "feature_contract": PHASE1_PROPOSAL_FEATURE_CONTRACT,
            "training_dataset_sha256": self.training_dataset_sha256,
            "thresholds": dict(self.thresholds),
            "minimum_development_precision": self.minimum_development_precision,
            "model": self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Phase1ProposalVerifier:
        if payload.get("schema_version") != _VERIFIER_SCHEMA:
            raise ValueError("Unsupported Phase 1 proposal verifier schema")
        if payload.get("feature_contract") != PHASE1_PROPOSAL_FEATURE_CONTRACT:
            raise ValueError("Proposal verifier feature contract is incompatible")
        raw_thresholds = payload.get("thresholds")
        raw_model = payload.get("model")
        if not isinstance(raw_thresholds, Mapping) or not isinstance(raw_model, Mapping):
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
        return cls(
            model=SparseLogisticModel.from_dict(raw_model),
            thresholds=tuple(sorted(thresholds)),
            training_dataset_sha256=str(payload.get("training_dataset_sha256", "")),
            minimum_development_precision=(
                float(payload["minimum_development_precision"])
                if isinstance(payload.get("minimum_development_precision"), int | float)
                and not isinstance(payload.get("minimum_development_precision"), bool)
                else None
            ),
        )


def fit_phase1_proposal_verifier(
    dataset: Phase1ProposalDataset,
    *,
    training_config: SparseLogisticTrainingConfig | None = None,
    minimum_development_precision: float | None = None,
) -> tuple[Phase1ProposalVerifier, dict[str, Any]]:
    """Fit on train, calibrate per-type thresholds on development, and report baselines."""

    if dataset.manifest.get("feature_contract") != PHASE1_PROPOSAL_FEATURE_CONTRACT:
        raise ValueError("Proposal dataset feature contract is incompatible")
    if minimum_development_precision is not None and not (
        0.0 < minimum_development_precision <= 1.0
    ):
        raise ValueError("Minimum development precision must be within (0, 1]")
    train = tuple(example for example in dataset.examples if example.split == "train")
    development = tuple(
        example for example in dataset.examples if example.split == "development"
    )
    if not train or not development:
        raise ValueError("Proposal calibration requires train and development examples")

    sparse_train = [
        SparseBinaryExample(
            features=example.features,
            label=example.label,
        )
        for example in train
    ]
    model, training_report = fit_sparse_logistic(
        sparse_train,
        config=training_config or SparseLogisticTrainingConfig(),
    )
    train_probabilities = tuple(
        model.predict_probability(dict(example.features)) for example in train
    )
    development_probabilities = tuple(
        model.predict_probability(dict(example.features)) for example in development
    )
    gold_counts = _gold_counts(dataset.manifest)
    thresholds, threshold_report = _calibrate_thresholds(
        development,
        development_probabilities,
        gold_counts,
        minimum_precision=minimum_development_precision,
    )
    dataset_sha256 = _mapping_sha256(dataset.manifest)
    verifier = Phase1ProposalVerifier(
        model=model,
        thresholds=tuple(sorted(thresholds.items())),
        training_dataset_sha256=dataset_sha256,
        minimum_development_precision=minimum_development_precision,
    )
    learned_selected = _select_examples(
        development,
        development_probabilities,
        thresholds,
        resolve_overlaps=True,
    )
    report = {
        "schema_version": "phase1-proposal-calibration-report.v1",
        "feature_contract": PHASE1_PROPOSAL_FEATURE_CONTRACT,
        "training_dataset_sha256": dataset_sha256,
        "holdout_opened": False,
        "operating_point": {
            "objective": (
                "maximum_recall_at_minimum_precision"
                if minimum_development_precision is not None
                else "maximum_f1"
            ),
            "minimum_development_precision": minimum_development_precision,
        },
        "training": training_report,
        "probability_metrics": {
            "train": binary_probability_metrics(
                [example.label for example in train],
                train_probabilities,
            ),
            "development": binary_probability_metrics(
                [example.label for example in development],
                development_probabilities,
            ),
        },
        "threshold_calibration": threshold_report,
        "development_selection": {
            "learned": _selection_metrics(
                learned_selected,
                gold_counts,
                split="development",
            ),
            **_development_baselines(development, gold_counts, dataset.manifest),
        },
        "coverage_ceiling": {
            split: _coverage_ceiling(dataset.examples, gold_counts, split=split)
            for split in ("train", "development")
        },
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
    thresholds = verifier.threshold_by_type
    for row in rows:
        document_id = str(row.get("document_id", ""))
        if document_id not in source_text_by_document:
            raise ValueError(f"Proposal references unknown document {document_id!r}")
        entity_type = str(row.get("type", ""))
        if entity_type not in thresholds:
            raise ValueError(f"Proposal has unsupported type {entity_type!r}")
        features = extract_phase1_proposal_features(
            row,
            source_text_by_document[document_id],
            source_roles,
            structure=structures[document_id],
        )
        probability = verifier.predict_probability(features)
        threshold = thresholds[entity_type]
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

    accepted_ids = {
        id(row)
        for row, _, _ in _resolve_runtime_overlaps(preselected)
    }
    return tuple(
        ScoredPhase1Proposal(
            row=item.row,
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
        )
        for item in scored
    )


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


def _select_examples(
    examples: Sequence[Phase1ProposalExample],
    probabilities: Sequence[float],
    thresholds: Mapping[str, float],
    *,
    resolve_overlaps: bool,
) -> tuple[Phase1ProposalExample, ...]:
    candidates = [
        (example, probability)
        for example, probability in zip(examples, probabilities, strict=True)
        if probability >= thresholds[example.entity_type]
    ]
    if not resolve_overlaps:
        return tuple(example for example, _ in candidates)
    accepted: list[tuple[Phase1ProposalExample, float]] = []
    for example, probability in sorted(candidates, key=_scored_example_sort_key):
        if any(
            example.document_id == other.document_id
            and _overlap(example.position, other.position)
            for other, _ in accepted
        ):
            continue
        accepted.append((example, probability))
    return tuple(example for example, _ in accepted)


def _resolve_runtime_overlaps(
    rows: Sequence[tuple[Mapping[str, Any], float, float]],
) -> tuple[tuple[Mapping[str, Any], float, float], ...]:
    accepted: list[tuple[Mapping[str, Any], float, float]] = []
    for item in sorted(rows, key=_runtime_score_sort_key):
        row = item[0]
        document_id = str(row.get("document_id", ""))
        position = _row_position(row)
        if any(
            document_id == str(other[0].get("document_id", ""))
            and _overlap(position, _row_position(other[0]))
            for other in accepted
        ):
            continue
        accepted.append(item)
    return tuple(accepted)


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
    true_positive = sum(example.label for example in scoped)
    false_positive = len(scoped) - true_positive
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
        "predicted": len(scoped),
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "selected_error_counts": dict(
            sorted(Counter(example.error_kind for example in scoped).items())
        ),
    }


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


def _scored_example_sort_key(
    item: tuple[Phase1ProposalExample, float],
) -> tuple[Any, ...]:
    example, probability = item
    return (
        -probability,
        -len(example.sources),
        -PHASE1_TYPE_PRIORITY.get(example.entity_type, 0),
        -(example.position[1] - example.position[0]),
        _document_sort_key(example.document_id),
        example.position,
        example.entity_type,
    )


def _runtime_score_sort_key(
    item: tuple[Mapping[str, Any], float, float],
) -> tuple[Any, ...]:
    row, probability, _ = item
    position = _row_position(row)
    raw_sources = row.get("sources")
    source_count = len(raw_sources) if isinstance(raw_sources, list) else 0
    entity_type = str(row.get("type", ""))
    return (
        -probability,
        -source_count,
        -PHASE1_TYPE_PRIORITY.get(entity_type, 0),
        -(position[1] - position[0]),
        _document_sort_key(str(row.get("document_id", ""))),
        position,
        entity_type,
    )


def _row_position(row: Mapping[str, Any]) -> tuple[int, int]:
    value = row.get("position")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ValueError("Proposal has an invalid position")
    return value[0], value[1]


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return min(left[1], right[1]) > max(left[0], right[0])


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
