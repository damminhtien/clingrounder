"""Dependency-free sparse logistic regression for calibrated decision gates.

The implementation is intentionally small: experiment gates need deterministic probabilities,
portable JSON weights, and fast CPU inference, not a second model-training framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "SparseBinaryExample",
    "SparseLogisticModel",
    "SparseLogisticTrainingConfig",
    "binary_probability_metrics",
    "fit_sparse_logistic",
]

_MODEL_SCHEMA = "sparse-logistic-model.v1"


@dataclass(frozen=True, slots=True)
class SparseBinaryExample:
    """One binary target and its finite, sparse numeric feature vector."""

    features: tuple[tuple[str, float], ...]
    label: int
    weight: float = 1.0

    @classmethod
    def from_mapping(
        cls,
        features: Mapping[str, float],
        *,
        label: int,
        weight: float = 1.0,
    ) -> SparseBinaryExample:
        """Canonicalize a feature mapping so training is order-independent."""

        if label not in {0, 1}:
            raise ValueError("Sparse binary labels must be zero or one")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("Sparse binary example weight must be finite and positive")
        canonical: list[tuple[str, float]] = []
        for name, value in sorted(features.items()):
            if not name:
                raise ValueError("Sparse feature names must be non-empty")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Sparse feature {name!r} must be finite")
            if numeric != 0.0:
                canonical.append((name, numeric))
        return cls(features=tuple(canonical), label=label, weight=weight)


@dataclass(frozen=True, slots=True)
class SparseLogisticTrainingConfig:
    """Deterministic full-batch optimizer settings."""

    epochs: int = 300
    learning_rate: float = 0.35
    learning_rate_decay: float = 0.02
    l2: float = 0.002
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("Logistic training epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("Logistic learning rate must be positive")
        if self.learning_rate_decay < 0.0:
            raise ValueError("Logistic learning-rate decay cannot be negative")
        if self.l2 < 0.0:
            raise ValueError("Logistic L2 penalty cannot be negative")
        if self.tolerance <= 0.0:
            raise ValueError("Logistic convergence tolerance must be positive")


@dataclass(frozen=True, slots=True)
class SparseLogisticModel:
    """Portable binary logistic model with a stable feature contract."""

    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.weights):
            raise ValueError("Logistic feature and weight dimensions differ")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("Logistic feature names must be unique")
        if tuple(sorted(self.feature_names)) != self.feature_names:
            raise ValueError("Logistic feature names must use canonical sorted order")
        if not math.isfinite(self.bias) or not all(
            math.isfinite(weight) for weight in self.weights
        ):
            raise ValueError("Logistic parameters must be finite")

    def predict_logit(self, features: Mapping[str, float]) -> float:
        """Return the linear score; unknown runtime features are ignored."""

        return self.bias + sum(
            weight * float(features.get(name, 0.0))
            for name, weight in zip(self.feature_names, self.weights, strict=True)
        )

    def predict_probability(self, features: Mapping[str, float]) -> float:
        """Return a numerically stable probability of the positive class."""

        return _sigmoid(self.predict_logit(features))

    def to_dict(self) -> dict[str, Any]:
        """Serialize without pickle so model artifacts remain inspectable."""

        return {
            "schema_version": _MODEL_SCHEMA,
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SparseLogisticModel:
        """Load and validate one inspectable model payload."""

        if payload.get("schema_version") != _MODEL_SCHEMA:
            raise ValueError("Unsupported sparse logistic model schema")
        names = payload.get("feature_names")
        weights = payload.get("weights")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("Logistic feature_names must be a string list")
        if not isinstance(weights, list) or not all(
            isinstance(weight, int | float) and not isinstance(weight, bool)
            for weight in weights
        ):
            raise ValueError("Logistic weights must be numeric")
        bias = payload.get("bias")
        if not isinstance(bias, int | float) or isinstance(bias, bool):
            raise ValueError("Logistic bias must be numeric")
        return cls(
            feature_names=tuple(names),
            weights=tuple(float(weight) for weight in weights),
            bias=float(bias),
        )


def fit_sparse_logistic(
    examples: Sequence[SparseBinaryExample],
    *,
    config: SparseLogisticTrainingConfig = SparseLogisticTrainingConfig(),
) -> tuple[SparseLogisticModel, dict[str, Any]]:
    """Fit regularized logistic regression with deterministic full-batch updates."""

    if not examples:
        raise ValueError("Logistic training requires examples")
    labels = {example.label for example in examples}
    if labels != {0, 1}:
        raise ValueError("Logistic training requires both binary classes")
    feature_names = tuple(
        sorted({name for example in examples for name, _ in example.features})
    )
    feature_index = {name: index for index, name in enumerate(feature_names)}
    encoded = tuple(
        (
            tuple((feature_index[name], value) for name, value in example.features),
            example.label,
            example.weight,
        )
        for example in examples
    )
    total_weight = sum(example.weight for example in examples)
    positive_rate = (
        sum(example.label * example.weight for example in examples) / total_weight
    )
    bias = math.log(_clamp_probability(positive_rate) / (1.0 - _clamp_probability(positive_rate)))
    weights = [0.0] * len(feature_names)
    iterations = 0
    converged = False
    maximum_delta = math.inf

    for epoch in range(config.epochs):
        gradients = [0.0] * len(weights)
        bias_gradient = 0.0
        for sparse, label, example_weight in encoded:
            logit = bias + sum(weights[index] * value for index, value in sparse)
            error = (_sigmoid(logit) - label) * example_weight
            bias_gradient += error
            for index, value in sparse:
                gradients[index] += error * value
        learning_rate = config.learning_rate / math.sqrt(
            1.0 + config.learning_rate_decay * epoch
        )
        bias_delta = learning_rate * bias_gradient / total_weight
        bias -= bias_delta
        maximum_delta = abs(bias_delta)
        for index, weight in enumerate(weights):
            gradient = gradients[index] / total_weight + config.l2 * weight
            delta = learning_rate * gradient
            weights[index] -= delta
            maximum_delta = max(maximum_delta, abs(delta))
        iterations = epoch + 1
        if maximum_delta < config.tolerance:
            converged = True
            break

    model = SparseLogisticModel(
        feature_names=feature_names,
        weights=tuple(weights),
        bias=bias,
    )
    probabilities = [
        model.predict_probability(dict(example.features)) for example in examples
    ]
    report = {
        "schema_version": "sparse-logistic-training-report.v1",
        "example_count": len(examples),
        "positive_count": sum(example.label for example in examples),
        "feature_count": len(feature_names),
        "iterations": iterations,
        "converged": converged,
        # MODEL: full-batch optimization can legitimately hit the configured iteration budget.
        # Persist the final update so downstream reports distinguish that case from a malformed
        # fit or a falsely assumed convergence.
        "stop_reason": "parameter_tolerance" if converged else "max_epochs",
        "final_maximum_parameter_update": maximum_delta,
        "config": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "learning_rate_decay": config.learning_rate_decay,
            "l2": config.l2,
            "tolerance": config.tolerance,
        },
        "metrics": binary_probability_metrics(
            [example.label for example in examples],
            probabilities,
        ),
    }
    return model, report


def binary_probability_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> dict[str, float | int]:
    """Report discrimination and calibration without third-party dependencies."""

    if len(labels) != len(probabilities) or not labels:
        raise ValueError("Probability metrics require equal non-empty inputs")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Probability threshold must be within [0, 1]")
    if calibration_bins < 1:
        raise ValueError("Calibration bin count must be positive")
    tp = fp = fn = tn = 0
    log_loss = 0.0
    brier = 0.0
    bins: list[list[tuple[int, float]]] = [[] for _ in range(calibration_bins)]
    for label, raw_probability in zip(labels, probabilities, strict=True):
        if label not in {0, 1} or not math.isfinite(raw_probability):
            raise ValueError("Probability metrics received invalid input")
        probability = _clamp_probability(raw_probability)
        predicted = int(probability >= threshold)
        tp += int(predicted == 1 and label == 1)
        fp += int(predicted == 1 and label == 0)
        fn += int(predicted == 0 and label == 1)
        tn += int(predicted == 0 and label == 0)
        log_loss -= label * math.log(probability) + (1 - label) * math.log(
            1.0 - probability
        )
        brier += (probability - label) ** 2
        bin_index = min(calibration_bins - 1, int(probability * calibration_bins))
        bins[bin_index].append((label, probability))
    count = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    calibration_error = sum(
        len(items)
        / count
        * abs(
            sum(probability for _, probability in items) / len(items)
            - sum(label for label, _ in items) / len(items)
        )
        for items in bins
        if items
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "log_loss": log_loss / count,
        "brier": brier / count,
        "expected_calibration_error": calibration_error,
    }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(max(value, -60.0))
    return exponent / (1.0 + exponent)


def _clamp_probability(value: float) -> float:
    return min(max(value, 1e-12), 1.0 - 1e-12)
