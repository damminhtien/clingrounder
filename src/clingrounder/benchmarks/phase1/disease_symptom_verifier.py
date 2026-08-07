"""Target-task verifier for ambiguous Phase 1 disease and symptom proposals."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from clingrounder.benchmarks.phase1.reviewed_corpus import (
    Phase1ReviewedCorpus,
)
from clingrounder.evaluation.sparse_logistic import (
    SparseBinaryExample,
    SparseLogisticModel,
    SparseLogisticTrainingConfig,
    fit_sparse_logistic,
)
from clingrounder.ner.document_structure import (
    DocumentStructure,
    DocumentStructureAnalyzer,
)
from clingrounder.utils.text import normalize_for_match

__all__ = [
    "DISEASE_SYMPTOM_FEATURE_CONTRACT",
    "DiseaseSymptomLabel",
    "DiseaseSymptomVerifier",
    "DiseaseSymptomVerifierDataset",
    "DiseaseSymptomVerifierExample",
    "build_disease_symptom_verifier_dataset",
    "extract_disease_symptom_features",
    "fit_disease_symptom_verifier",
    "write_disease_symptom_verifier",
]

DISEASE_SYMPTOM_FEATURE_CONTRACT = "phase1-disease-symptom-features.v1"
_DATASET_SCHEMA = "phase1-disease-symptom-dataset.v1"
_VERIFIER_SCHEMA = "phase1-disease-symptom-verifier.v1"
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_QUESTION_RE = re.compile(
    r"^[ \t]*(?:câu[ \t]+hỏi|hỏi|question)[ \t]*(?::|-)",
    flags=re.IGNORECASE | re.UNICODE,
)
_ANSWER_RE = re.compile(
    r"^[ \t]*(?:đáp[ \t]+án|trả[ \t]+lời|answer)[ \t]*(?::|-)",
    flags=re.IGNORECASE | re.UNICODE,
)
_HASH_BUCKETS = 512
_HARD_CASE_MENTIONS = (
    "chóng mặt",
    "mất ngủ",
    "suy nhược",
    "đau ngực",
    "thiếu máu",
)


class DiseaseSymptomLabel(StrEnum):
    """The only target labels emitted by the dedicated verifier."""

    DISEASE = "DISEASE"
    SYMPTOM = "SYMPTOM"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class DiseaseSymptomVerifierExample:
    """One raw mention with a target-task label and inspectable evidence."""

    document_id: str
    split: str
    text: str
    position: tuple[int, int]
    label: DiseaseSymptomLabel
    reason: str
    representation_labels: tuple[str, ...]
    features: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "split": self.split,
            "text": self.text,
            "position": list(self.position),
            "label": self.label.value,
            "reason": self.reason,
            "representation_labels": list(self.representation_labels),
            "features": dict(self.features),
        }


@dataclass(frozen=True, slots=True)
class DiseaseSymptomVerifierDataset:
    """Leakage-safe examples and their immutable build manifest."""

    examples: tuple[DiseaseSymptomVerifierExample, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DiseaseSymptomVerifier:
    """Portable three-way classifier with conservative target-label abstention."""

    models: tuple[tuple[DiseaseSymptomLabel, SparseLogisticModel], ...]
    disease_threshold: float
    symptom_threshold: float
    minimum_margin: float
    training_dataset_sha256: str

    def __post_init__(self) -> None:
        labels = {label for label, _ in self.models}
        if labels != set(DiseaseSymptomLabel) or len(labels) != len(self.models):
            raise ValueError("Type verifier must contain one model per target class")
        if not 0.0 <= self.disease_threshold <= 1.0:
            raise ValueError("Disease threshold must be within [0, 1]")
        if not 0.0 <= self.symptom_threshold <= 1.0:
            raise ValueError("Symptom threshold must be within [0, 1]")
        if not 0.0 <= self.minimum_margin <= 1.0:
            raise ValueError("Verifier margin must be within [0, 1]")
        if len(self.training_dataset_sha256) != 64:
            raise ValueError("Type verifier requires a dataset SHA-256")

    @property
    def model_by_label(self) -> dict[DiseaseSymptomLabel, SparseLogisticModel]:
        return dict(self.models)

    def predict_probabilities(
        self,
        features: Mapping[str, float],
    ) -> dict[DiseaseSymptomLabel, float]:
        """Return normalized one-vs-rest logits for all three classes."""

        logits = {
            label: model.predict_logit(features)
            for label, model in self.models
        }
        maximum = max(logits.values())
        exponents = {
            label: math.exp(min(60.0, value - maximum))
            for label, value in logits.items()
        }
        denominator = sum(exponents.values())
        return {
            label: value / denominator for label, value in exponents.items()
        }

    def predict(self, features: Mapping[str, float]) -> DiseaseSymptomLabel:
        """Classify or abstain; ``DISEASE`` is never a fallback label."""

        probabilities = self.predict_probabilities(features)
        ranked = sorted(
            probabilities.items(),
            key=lambda item: (item[1], item[0].value),
            reverse=True,
        )
        best_label, best_probability = ranked[0]
        margin = best_probability - ranked[1][1]
        if best_label is DiseaseSymptomLabel.NONE:
            return DiseaseSymptomLabel.NONE
        threshold = (
            self.disease_threshold
            if best_label is DiseaseSymptomLabel.DISEASE
            else self.symptom_threshold
        )
        if best_probability < threshold or margin < self.minimum_margin:
            return DiseaseSymptomLabel.NONE
        return best_label

    def classify(
        self,
        source_text: str,
        position: tuple[int, int],
        *,
        representation_labels: Sequence[str] = (),
        structure: DocumentStructure | None = None,
    ) -> DiseaseSymptomLabel:
        """Extract target-task features and classify one immutable raw span."""

        features = extract_disease_symptom_features(
            source_text,
            position,
            representation_labels=representation_labels,
            structure=structure,
        )
        return self.predict(features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _VERIFIER_SCHEMA,
            "feature_contract": DISEASE_SYMPTOM_FEATURE_CONTRACT,
            "training_dataset_sha256": self.training_dataset_sha256,
            "operating_point": {
                "disease_threshold": self.disease_threshold,
                "symptom_threshold": self.symptom_threshold,
                "minimum_margin": self.minimum_margin,
                "default_label": DiseaseSymptomLabel.NONE.value,
                "disease_fallback": False,
            },
            "models": {
                label.value: model.to_dict() for label, model in self.models
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DiseaseSymptomVerifier:
        if payload.get("schema_version") != _VERIFIER_SCHEMA:
            raise ValueError("Unsupported disease/symptom verifier schema")
        if payload.get("feature_contract") != DISEASE_SYMPTOM_FEATURE_CONTRACT:
            raise ValueError("Disease/symptom feature contract is incompatible")
        raw_models = payload.get("models")
        operating_point = payload.get("operating_point")
        if not isinstance(raw_models, Mapping) or not isinstance(
            operating_point, Mapping
        ):
            raise ValueError("Disease/symptom verifier artifact is incomplete")
        if operating_point.get("disease_fallback") is not False:
            raise ValueError("Disease fallback is forbidden")
        models = tuple(
            sorted(
                (
                    DiseaseSymptomLabel(str(label)),
                    SparseLogisticModel.from_dict(model),
                )
                for label, model in raw_models.items()
                if isinstance(model, Mapping)
            )
        )
        return cls(
            models=models,
            disease_threshold=float(operating_point["disease_threshold"]),
            symptom_threshold=float(operating_point["symptom_threshold"]),
            minimum_margin=float(operating_point["minimum_margin"]),
            training_dataset_sha256=str(payload.get("training_dataset_sha256", "")),
        )


def extract_disease_symptom_features(
    source_text: str,
    position: tuple[int, int],
    *,
    representation_labels: Sequence[str] = (),
    structure: DocumentStructure | None = None,
) -> dict[str, float]:
    """Encode mention, local context, section, Q/A role, and model evidence.

    MODEL: broad labels such as VietMed ``DISEASESYMTOM`` are representation features only. They
    are intentionally absent from the target-label construction path.
    """

    start, end = position
    if start < 0 or end <= start or end > len(source_text):
        raise ValueError("Type-verifier position is outside source text")
    mention = source_text[start:end]
    if not mention:
        raise ValueError("Type-verifier mention must be non-empty")
    active_structure = structure or DocumentStructureAnalyzer().analyze(source_text)
    normalized = normalize_for_match(mention)
    words = _WORD_RE.findall(normalized)
    features: dict[str, float] = {
        "bias:present": 1.0,
        "numeric:span_log_length": math.log1p(end - start),
        "numeric:token_log_count": math.log1p(len(words)),
        f"genre:{active_structure.genre.value}": 1.0,
        f"qa_role:{_question_answer_role(source_text, start)}": 1.0,
    }
    section = active_structure.section_at(start)
    section_name = section.kind.value if section is not None else "none"
    features[f"section:{section_name}"] = 1.0
    features["flag:single_token"] = float(len(words) == 1)
    features["flag:contains_digit"] = float(any(char.isdigit() for char in mention))
    features["flag:starts_upper"] = float(
        bool(mention.strip()) and mention.strip()[0].isupper()
    )
    padded = f"^{normalized}$"
    for size in (2, 3, 4):
        for index in range(max(0, len(padded) - size + 1)):
            _add_hash(
                features,
                f"mention_char_{size}",
                padded[index : index + size],
            )
    left_words = _WORD_RE.findall(
        normalize_for_match(source_text[max(0, start - 128) : start])
    )
    right_words = _WORD_RE.findall(
        normalize_for_match(source_text[end : min(len(source_text), end + 128)])
    )
    for distance, word in enumerate(reversed(left_words[-6:]), start=1):
        _add_hash(features, f"context_left_{distance}", word)
    for distance, word in enumerate(right_words[:6], start=1):
        _add_hash(features, f"context_right_{distance}", word)
    for label in sorted(set(representation_labels)):
        _add_hash(features, "representation_label", label)
    return dict(sorted(features.items()))


def build_disease_symptom_verifier_dataset(
    corpus: Phase1ReviewedCorpus,
    *,
    proposal_matrix_path: str | Path,
    corpus_fingerprint_sha256: str,
    representation_rows_by_document: (
        Mapping[str, Sequence[Mapping[str, Any]]] | None
    ) = None,
) -> DiseaseSymptomVerifierDataset:
    """Build train/development examples without opening holdout labels."""

    selected_ids = set(corpus.split_by_document)
    proposals = _read_jsonl(Path(proposal_matrix_path))
    proposal_positions: dict[str, set[tuple[int, int]]] = {
        document_id: set() for document_id in selected_ids
    }
    for row in proposals:
        document_id = str(row.get("document_id", ""))
        if document_id not in selected_ids:
            continue
        position = _position(row)
        source_text = corpus.source_texts[document_id]
        if source_text[position[0] : position[1]] != row.get("text"):
            raise ValueError(
                f"Proposal offset mismatch in document {document_id}: {position}"
            )
        proposal_positions[document_id].add(position)

    representation = _representation_by_span(
        representation_rows_by_document or {},
        corpus.source_texts,
        selected_ids,
    )
    analyzer = DocumentStructureAnalyzer()
    examples: list[DiseaseSymptomVerifierExample] = []
    for document_id in sorted(selected_ids, key=_document_sort_key):
        source_text = corpus.source_texts[document_id]
        gold = corpus.gold_rows[document_id]
        gold_labels = _gold_labels_by_span(gold, source_text)
        positions = set(gold_labels) | proposal_positions[document_id]
        structure = analyzer.analyze(source_text)
        for position in sorted(positions):
            label, reason = _example_label(position, gold_labels, gold)
            representation_labels = tuple(
                sorted(representation.get((document_id, *position), set()))
            )
            features = extract_disease_symptom_features(
                source_text,
                position,
                representation_labels=representation_labels,
                structure=structure,
            )
            examples.append(
                DiseaseSymptomVerifierExample(
                    document_id=document_id,
                    split=corpus.split_by_document[document_id],
                    text=source_text[position[0] : position[1]],
                    position=position,
                    label=label,
                    reason=reason,
                    representation_labels=representation_labels,
                    features=tuple(features.items()),
                )
            )
    examples.sort(key=_example_sort_key)
    serialized = _serialize_examples(examples)
    counts = Counter(
        f"{example.split}:{example.label.value}" for example in examples
    )
    reasons = Counter(f"{example.split}:{example.reason}" for example in examples)
    representation_counts = Counter(
        label
        for example in examples
        for label in example.representation_labels
    )
    manifest = {
        "schema_version": _DATASET_SCHEMA,
        "feature_contract": DISEASE_SYMPTOM_FEATURE_CONTRACT,
        "example_count": len(examples),
        "examples_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "corpus_fingerprint_sha256": corpus_fingerprint_sha256,
        "proposal_matrix_sha256": _sha256_file(proposal_matrix_path),
        "split_label_counts": dict(sorted(counts.items())),
        "split_reason_counts": dict(sorted(reasons.items())),
        "representation_label_counts": dict(sorted(representation_counts.items())),
        "holdout_opened": False,
        "target_policy": {
            "labels": [label.value for label in DiseaseSymptomLabel],
            "default_label": DiseaseSymptomLabel.NONE.value,
            "disease_fallback": False,
            "representation_labels_are_targets": False,
        },
    }
    return DiseaseSymptomVerifierDataset(
        examples=tuple(examples),
        manifest=manifest,
    )


def fit_disease_symptom_verifier(
    dataset: DiseaseSymptomVerifierDataset,
    *,
    training_config: SparseLogisticTrainingConfig | None = None,
) -> tuple[DiseaseSymptomVerifier, dict[str, Any]]:
    """Fit three one-vs-rest models and calibrate an abstaining operating point."""

    if dataset.manifest.get("feature_contract") != DISEASE_SYMPTOM_FEATURE_CONTRACT:
        raise ValueError("Disease/symptom dataset feature contract is incompatible")
    train = tuple(example for example in dataset.examples if example.split == "train")
    development = tuple(
        example for example in dataset.examples if example.split == "development"
    )
    if not train or not development:
        raise ValueError("Type verifier requires train and development examples")
    models: list[tuple[DiseaseSymptomLabel, SparseLogisticModel]] = []
    training_reports: dict[str, Any] = {}
    for label in DiseaseSymptomLabel:
        binary = _binary_examples(train, label)
        model, report = fit_sparse_logistic(
            binary,
            config=training_config or SparseLogisticTrainingConfig(),
        )
        models.append((label, model))
        training_reports[label.value] = report
    temporary = DiseaseSymptomVerifier(
        models=tuple(sorted(models)),
        disease_threshold=0.0,
        symptom_threshold=0.0,
        minimum_margin=0.0,
        training_dataset_sha256=str(dataset.manifest["examples_sha256"]),
    )
    development_probabilities = [
        temporary.predict_probabilities(dict(example.features))
        for example in development
    ]
    disease_threshold, symptom_threshold, margin, search = _calibrate_operating_point(
        development,
        development_probabilities,
    )
    verifier = DiseaseSymptomVerifier(
        models=temporary.models,
        disease_threshold=disease_threshold,
        symptom_threshold=symptom_threshold,
        minimum_margin=margin,
        training_dataset_sha256=temporary.training_dataset_sha256,
    )
    train_predictions = [
        verifier.predict(dict(example.features)) for example in train
    ]
    development_predictions = [
        verifier.predict(dict(example.features)) for example in development
    ]
    report = {
        "schema_version": "phase1-disease-symptom-calibration-report.v1",
        "holdout_opened": False,
        "training_dataset_sha256": verifier.training_dataset_sha256,
        "training": training_reports,
        "operating_point_search": search,
        "metrics": {
            "train": _classification_metrics(train, train_predictions),
            "development": _classification_metrics(
                development,
                development_predictions,
            ),
        },
        "hard_case_metrics": {
            "train": _hard_case_metrics(train, train_predictions),
            "development": _hard_case_metrics(
                development,
                development_predictions,
            ),
        },
        "policy": {
            "disease_fallback": False,
            "default_label": DiseaseSymptomLabel.NONE.value,
            "vietmed_diseasesymtom_is_target": False,
        },
    }
    return verifier, report


def write_disease_symptom_verifier(
    dataset: DiseaseSymptomVerifierDataset,
    verifier: DiseaseSymptomVerifier,
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    """Write deterministic training rows, model weights, and calibration metrics."""

    output = Path(output_dir)
    dataset_dir = output / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(dataset.manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "examples.jsonl").write_text(
        _serialize_examples(dataset.examples),
        encoding="utf-8",
    )
    (output / "verifier.json").write_text(
        json.dumps(verifier.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _binary_examples(
    examples: Sequence[DiseaseSymptomVerifierExample],
    positive_label: DiseaseSymptomLabel,
) -> list[SparseBinaryExample]:
    positive_count = sum(example.label is positive_label for example in examples)
    negative_count = len(examples) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError(f"Class {positive_label.value} lacks binary training support")
    positive_weight = len(examples) / (2.0 * positive_count)
    negative_weight = len(examples) / (2.0 * negative_count)
    return [
        SparseBinaryExample(
            features=example.features,
            label=int(example.label is positive_label),
            weight=(
                positive_weight
                if example.label is positive_label
                else negative_weight
            ),
        )
        for example in examples
    ]


def _calibrate_operating_point(
    examples: Sequence[DiseaseSymptomVerifierExample],
    probabilities: Sequence[Mapping[DiseaseSymptomLabel, float]],
) -> tuple[float, float, float, dict[str, Any]]:
    thresholds = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)
    margins = (0.0, 0.03, 0.05, 0.08, 0.10, 0.15)
    trials: list[dict[str, Any]] = []
    for disease_threshold in thresholds:
        for symptom_threshold in thresholds:
            for margin in margins:
                predictions = [
                    _decision(
                        row,
                        disease_threshold=disease_threshold,
                        symptom_threshold=symptom_threshold,
                        minimum_margin=margin,
                    )
                    for row in probabilities
                ]
                metrics = _classification_metrics(examples, predictions)
                trials.append(
                    {
                        "disease_threshold": disease_threshold,
                        "symptom_threshold": symptom_threshold,
                        "minimum_margin": margin,
                        "metrics": metrics,
                    }
                )
    best = max(
        trials,
        key=lambda row: (
            float(row["metrics"]["macro_f1"]),
            float(row["metrics"]["target_macro_f1"]),
            float(row["metrics"]["accuracy"]),
            float(row["disease_threshold"]) + float(row["symptom_threshold"]),
            float(row["minimum_margin"]),
        ),
    )
    return (
        float(best["disease_threshold"]),
        float(best["symptom_threshold"]),
        float(best["minimum_margin"]),
        {
            "objective": "macro_f1_then_target_macro_f1",
            "trial_count": len(trials),
            "selected": best,
        },
    )


def _decision(
    probabilities: Mapping[DiseaseSymptomLabel, float],
    *,
    disease_threshold: float,
    symptom_threshold: float,
    minimum_margin: float,
) -> DiseaseSymptomLabel:
    ranked = sorted(
        probabilities.items(),
        key=lambda item: (item[1], item[0].value),
        reverse=True,
    )
    label, probability = ranked[0]
    if label is DiseaseSymptomLabel.NONE:
        return label
    threshold = (
        disease_threshold
        if label is DiseaseSymptomLabel.DISEASE
        else symptom_threshold
    )
    return (
        label
        if probability >= threshold
        and probability - ranked[1][1] >= minimum_margin
        else DiseaseSymptomLabel.NONE
    )


def _classification_metrics(
    examples: Sequence[DiseaseSymptomVerifierExample],
    predictions: Sequence[DiseaseSymptomLabel],
) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("Classification metrics require aligned inputs")
    confusion = {
        gold.value: {predicted.value: 0 for predicted in DiseaseSymptomLabel}
        for gold in DiseaseSymptomLabel
    }
    for example, prediction in zip(examples, predictions, strict=True):
        confusion[example.label.value][prediction.value] += 1
    per_class: dict[str, Any] = {}
    for label in DiseaseSymptomLabel:
        tp = confusion[label.value][label.value]
        fp = sum(
            confusion[other.value][label.value]
            for other in DiseaseSymptomLabel
            if other is not label
        )
        fn = sum(
            confusion[label.value][other.value]
            for other in DiseaseSymptomLabel
            if other is not label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label.value] = {
            "support": sum(confusion[label.value].values()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    macro_f1 = sum(
        float(per_class[label.value]["f1"]) for label in DiseaseSymptomLabel
    ) / len(DiseaseSymptomLabel)
    target_macro_f1 = (
        float(per_class[DiseaseSymptomLabel.DISEASE.value]["f1"])
        + float(per_class[DiseaseSymptomLabel.SYMPTOM.value]["f1"])
    ) / 2.0
    correct = sum(
        confusion[label.value][label.value] for label in DiseaseSymptomLabel
    )
    return {
        "example_count": len(examples),
        "accuracy": correct / len(examples) if examples else 0.0,
        "macro_f1": macro_f1,
        "target_macro_f1": target_macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _hard_case_metrics(
    examples: Sequence[DiseaseSymptomVerifierExample],
    predictions: Sequence[DiseaseSymptomLabel],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for mention in _HARD_CASE_MENTIONS:
        indices = [
            index
            for index, example in enumerate(examples)
            if normalize_for_match(example.text) == mention
        ]
        rows[mention] = {
            "support": len(indices),
            "gold_counts": dict(
                sorted(Counter(examples[index].label.value for index in indices).items())
            ),
            "predicted_counts": dict(
                sorted(Counter(predictions[index].value for index in indices).items())
            ),
            "correct": sum(
                examples[index].label is predictions[index] for index in indices
            ),
        }
    return rows


def _gold_labels_by_span(
    rows: Sequence[Mapping[str, Any]],
    source_text: str,
) -> dict[tuple[int, int], DiseaseSymptomLabel]:
    labels: dict[tuple[int, int], DiseaseSymptomLabel] = {}
    for row in rows:
        position = _position(row)
        text = row.get("text")
        if not isinstance(text, str) or source_text[position[0] : position[1]] != text:
            raise ValueError("Gold text does not match raw source offset")
        entity_type = str(row.get("type", ""))
        label = (
            DiseaseSymptomLabel.DISEASE
            if entity_type == "CHẨN_ĐOÁN"
            else (
                DiseaseSymptomLabel.SYMPTOM
                if entity_type == "TRIỆU_CHỨNG"
                else DiseaseSymptomLabel.NONE
            )
        )
        previous = labels.get(position)
        if previous is not None and previous is not label:
            raise ValueError(f"Conflicting gold target labels at {position}")
        labels[position] = label
    return labels


def _example_label(
    position: tuple[int, int],
    gold_labels: Mapping[tuple[int, int], DiseaseSymptomLabel],
    gold_rows: Sequence[Mapping[str, Any]],
) -> tuple[DiseaseSymptomLabel, str]:
    exact = gold_labels.get(position)
    if exact is not None:
        return exact, (
            "gold_target"
            if exact is not DiseaseSymptomLabel.NONE
            else "gold_other_type"
        )
    target_overlap = any(
        _overlap(position, _position(row))
        and str(row.get("type")) in {"CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}
        for row in gold_rows
    )
    return (
        DiseaseSymptomLabel.NONE,
        "boundary_hard_negative" if target_overlap else "spurious_hard_negative",
    )


def _representation_by_span(
    rows_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    source_text_by_document: Mapping[str, str],
    selected_ids: set[str],
) -> dict[tuple[str, int, int], set[str]]:
    representation: dict[tuple[str, int, int], set[str]] = {}
    for document_id, rows in rows_by_document.items():
        if document_id not in selected_ids:
            continue
        source_text = source_text_by_document[document_id]
        for row in rows:
            position = _position(row)
            text = row.get("text")
            if not isinstance(text, str) or source_text[position[0] : position[1]] != text:
                raise ValueError("Representation row does not match raw source offset")
            source_label = str(row.get("source_label", ""))
            if source_label:
                representation.setdefault((document_id, *position), set()).add(
                    source_label
                )
    return representation


def _question_answer_role(source_text: str, start: int) -> str:
    line_start = source_text.rfind("\n", 0, start) + 1
    line_end = source_text.find("\n", start)
    if line_end < 0:
        line_end = len(source_text)
    current = source_text[line_start:line_end]
    if _QUESTION_RE.match(current):
        return "question"
    if _ANSWER_RE.match(current):
        return "answer"
    role = "none"
    for line in source_text[:start].splitlines():
        if _QUESTION_RE.match(line):
            role = "question"
        elif _ANSWER_RE.match(line):
            role = "answer"
    return role


def _add_hash(features: dict[str, float], namespace: str, value: str) -> None:
    digest = hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % _HASH_BUCKETS
    features[f"hash:{namespace}:{bucket:03d}"] = 1.0


def _position(row: Mapping[str, Any]) -> tuple[int, int]:
    value = row.get("position")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
        or value[0] >= value[1]
    ):
        raise ValueError("Entity row has an invalid position")
    return value[0], value[1]


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(payload)
    return rows


def _serialize_examples(examples: Sequence[DiseaseSymptomVerifierExample]) -> str:
    return "".join(
        json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for example in examples
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _example_sort_key(
    example: DiseaseSymptomVerifierExample,
) -> tuple[Any, ...]:
    return (
        _document_sort_key(example.document_id),
        example.position[0],
        example.position[1],
    )


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
