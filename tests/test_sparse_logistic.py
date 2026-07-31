"""Contracts for the dependency-free calibrated decision model."""

from __future__ import annotations

import pytest

from medical_kg_nlp.evaluation.sparse_logistic import (
    SparseBinaryExample,
    SparseLogisticModel,
    binary_probability_metrics,
    fit_sparse_logistic,
)


def test_sparse_logistic_learns_and_round_trips() -> None:
    examples = [
        SparseBinaryExample.from_mapping({"agreement": 1.0}, label=1),
        SparseBinaryExample.from_mapping({"agreement": 1.0, "long": 1.0}, label=1),
        SparseBinaryExample.from_mapping({"conflict": 1.0}, label=0),
        SparseBinaryExample.from_mapping({"conflict": 1.0, "long": 1.0}, label=0),
    ]

    model, report = fit_sparse_logistic(examples)
    restored = SparseLogisticModel.from_dict(model.to_dict())

    assert restored.predict_probability({"agreement": 1.0}) > 0.8
    assert restored.predict_probability({"conflict": 1.0}) < 0.2
    assert restored.predict_probability({"agreement": 1.0}) == pytest.approx(
        model.predict_probability({"agreement": 1.0})
    )
    assert report["example_count"] == 4
    assert report["metrics"]["f1"] == 1.0
    assert report["stop_reason"] in {"parameter_tolerance", "max_epochs"}
    assert report["final_maximum_parameter_update"] >= 0.0


def test_sparse_logistic_rejects_single_class_training() -> None:
    examples = [
        SparseBinaryExample.from_mapping({"source": 1.0}, label=1),
    ]

    with pytest.raises(ValueError, match="both binary classes"):
        fit_sparse_logistic(examples)


def test_binary_probability_metrics_reports_calibration() -> None:
    metrics = binary_probability_metrics(
        [1, 1, 0, 0],
        [0.9, 0.8, 0.2, 0.1],
    )

    assert metrics["f1"] == 1.0
    assert metrics["brier"] == pytest.approx(0.025)
    assert metrics["expected_calibration_error"] == pytest.approx(0.15)
