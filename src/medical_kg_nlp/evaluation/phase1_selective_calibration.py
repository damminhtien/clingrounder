from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.ontology.phase1 import (
    PHASE1_ASSERTABLE_TYPES,
    PHASE1_CODABLE_TYPES,
    PHASE1_TYPE_BY_ENTITY_TYPE,
    expected_code_system,
)
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import CodeSystem
from medical_kg_nlp.schema.types import AssertionStatus
from medical_kg_nlp.utils.text import normalize_for_match


@dataclass(frozen=True)
class CandidateCalibrationOptions:
    folds: int = 5
    minimum_support: int = 5
    minimum_train_support: int = 4
    abstention_margin: float = 0.05
    maximum_fold_regression: float = 0.02
    beta_alpha: float = 1.0
    beta_beta: float = 1.0

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        if self.minimum_support < 1 or self.minimum_train_support < 1:
            raise ValueError("support thresholds must be positive")
        for name, value in (
            ("abstention_margin", self.abstention_margin),
            ("maximum_fold_regression", self.maximum_fold_regression),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.beta_alpha <= 0.0 or self.beta_beta <= 0.0:
            raise ValueError("beta prior values must be positive")


@dataclass(frozen=True)
class _CandidateObservation:
    document_id: str
    fold: int
    entity_type: str
    code_system: str
    source: str
    evidence_sources: tuple[str, ...]
    mention_structure: str
    reviewed: bool
    reviewed_folds: frozenset[int]
    code: str
    retrieval_score: float
    emitted_score: float
    abstained_score: float
    correct: bool

    @property
    def source_key(self) -> tuple[str, str]:
        return self.code_system, self.source

    @property
    def policy_key(self) -> tuple[str, str, str, bool, str]:
        return (
            self.entity_type,
            self.code_system,
            self.source,
            self.reviewed,
            self.mention_structure,
        )


@dataclass(frozen=True)
class _AssertionObservation:
    document_id: str
    fold: int
    assertion: str
    entity_type: str
    scope: str
    rule_id: str
    cue: str
    emitted_score: float
    abstained_score: float
    correct: bool


def build_candidate_calibration_report(
    predictions: Iterable[ClinicalPrediction],
    gold_by_document: Mapping[str, list[dict[str, Any]]],
    *,
    reviewed_candidates: frozenset[tuple[str, str, str]] = frozenset(),
    options: CandidateCalibrationOptions = CandidateCalibrationOptions(),
) -> dict[str, Any]:
    prediction_by_document = {prediction.document_id: prediction for prediction in predictions}
    cross_fitted_reviewed = _cross_fitted_reviewed_candidates(
        gold_by_document,
        reviewed_candidates,
        folds=options.folds,
    )
    observations: list[_CandidateObservation] = []
    counters: Counter[str] = Counter()
    matched_opportunities: Counter[str] = Counter()

    for document_id in sorted(gold_by_document, key=_document_sort_key):
        prediction = prediction_by_document.get(document_id)
        if prediction is None:
            counters["missing_prediction_document"] += 1
            continue
        gold_index = _gold_index(gold_by_document[document_id], counters)
        for entity in prediction.entities:
            phase1_type = PHASE1_TYPE_BY_ENTITY_TYPE.get(entity.type)
            if phase1_type not in PHASE1_CODABLE_TYPES:
                continue
            key = (entity.span[0], entity.span[1], phase1_type)
            gold = gold_index.get(key)
            if gold is None:
                counters["entity_without_exact_gold_match"] += 1
                continue
            matched_opportunities[phase1_type] += 1
            expected_system = expected_code_system(phase1_type)
            top = _top_candidate(entity, expected_system)
            if top is None:
                counters["matched_entity_without_qualified_candidate"] += 1
                continue
            gold_codes = _candidate_set(gold.get("candidates"))
            emitted_score = (1.0 / len(gold_codes)) if top.code in gold_codes else 0.0
            abstained_score = 1.0 if not gold_codes else 0.0
            normalized = normalize_for_match(entity.text)
            reviewed_key = (normalized, phase1_type, str(top.code))
            observations.append(
                _CandidateObservation(
                    document_id=document_id,
                    fold=_fold(document_id, options.folds),
                    entity_type=phase1_type,
                    code_system=top.code_system.value,
                    source=top.source,
                    evidence_sources=top.evidence_sources,
                    mention_structure=_mention_structure(entity, top.code_system),
                    reviewed=reviewed_key in reviewed_candidates,
                    reviewed_folds=frozenset(
                        fold
                        for fold, candidates in cross_fitted_reviewed.items()
                        if reviewed_key in candidates
                    ),
                    code=str(top.code),
                    retrieval_score=top.retrieval_score,
                    emitted_score=emitted_score,
                    abstained_score=abstained_score,
                    correct=top.code in gold_codes,
                )
            )

    source_groups = _group_report(
        observations,
        key=lambda item: item.source_key,
        key_fields=("code_system", "source"),
        options=options,
    )
    reviewed_source_groups = _group_report(
        [item for item in observations if item.reviewed],
        key=lambda item: item.source_key,
        key_fields=("code_system", "source"),
        options=options,
    )
    policy_groups = _group_report(
        observations,
        key=lambda item: item.policy_key,
        key_fields=(
            "entity_type",
            "code_system",
            "source",
            "reviewed",
            "mention_structure",
        ),
        options=options,
    )
    recommended_source_probabilities = {
        f"{group['code_system']}:{group['source']}": group["smoothed_correct_probability"]
        for group in reviewed_source_groups
        if group["recommended"]
    }
    return {
        "schema_version": "phase1-candidate-calibration.v1",
        "options": {
            "folds": options.folds,
            "minimum_support": options.minimum_support,
            "minimum_train_support": options.minimum_train_support,
            "abstention_margin": options.abstention_margin,
            "maximum_fold_regression": options.maximum_fold_regression,
            "beta_alpha": options.beta_alpha,
            "beta_beta": options.beta_beta,
        },
        "coverage": {
            "gold_document_count": len(gold_by_document),
            "prediction_document_count": len(prediction_by_document),
            "matched_candidate_opportunities": dict(sorted(matched_opportunities.items())),
            "observation_count": len(observations),
            "reviewed_observation_count": sum(item.reviewed for item in observations),
            "counters": dict(sorted(counters.items())),
        },
        "source_groups": source_groups,
        "reviewed_source_groups": reviewed_source_groups,
        "policy_groups": policy_groups,
        "recommended_link_emit_probabilities_by_source": dict(
            sorted(recommended_source_probabilities.items())
        ),
    }


def build_assertion_calibration_report(
    predictions: Iterable[ClinicalPrediction],
    gold_by_document: Mapping[str, list[dict[str, Any]]],
    *,
    options: CandidateCalibrationOptions = CandidateCalibrationOptions(),
) -> dict[str, Any]:
    status_to_label = {
        AssertionStatus.NEGATED: "isNegated",
        AssertionStatus.FAMILY: "isFamily",
        AssertionStatus.HISTORICAL: "isHistorical",
    }
    prediction_by_document = {prediction.document_id: prediction for prediction in predictions}
    observations: list[_AssertionObservation] = []
    counters: Counter[str] = Counter()
    for document_id in sorted(gold_by_document, key=_document_sort_key):
        prediction = prediction_by_document.get(document_id)
        if prediction is None:
            counters["missing_prediction_document"] += 1
            continue
        gold_index = _gold_index(gold_by_document[document_id], counters)
        for entity in prediction.entities:
            phase1_type = PHASE1_TYPE_BY_ENTITY_TYPE.get(entity.type)
            if phase1_type not in PHASE1_ASSERTABLE_TYPES:
                continue
            gold = gold_index.get((entity.span[0], entity.span[1], phase1_type))
            if gold is None:
                counters["entity_without_exact_gold_match"] += 1
                continue
            gold_assertions = _candidate_set(gold.get("assertions"))
            seen: set[tuple[str, str]] = set()
            for evidence in entity.assertion_evidence:
                label = status_to_label.get(evidence.assertion)
                if label is None or (label, evidence.rule_id) in seen:
                    continue
                seen.add((label, evidence.rule_id))
                observations.append(
                    _AssertionObservation(
                        document_id=document_id,
                        fold=_fold(document_id, options.folds),
                        assertion=label,
                        entity_type=phase1_type,
                        scope=evidence.scope,
                        rule_id=evidence.rule_id,
                        cue=evidence.cue,
                        emitted_score=(
                            1.0 / len(gold_assertions)
                            if label in gold_assertions
                            else 0.0
                        ),
                        abstained_score=1.0 if not gold_assertions else 0.0,
                        correct=label in gold_assertions,
                    )
                )
    grouped: dict[tuple[str, str, str, str, str], list[_AssertionObservation]] = defaultdict(list)
    for observation in observations:
        grouped[
            (
                observation.assertion,
                observation.entity_type,
                observation.scope,
                observation.rule_id,
                observation.cue,
            )
        ].append(observation)
    groups: list[dict[str, Any]] = []
    for group_key, values in sorted(grouped.items()):
        support = len(values)
        correct = sum(item.correct for item in values)
        emitted_total = sum(item.emitted_score for item in values)
        abstained_total = sum(item.abstained_score for item in values)
        smoothed_emit = _smoothed(emitted_total, support, options)
        smoothed_abstain = _smoothed(abstained_total, support, options)
        cross_validation = _cross_validate(
            values,
            options,
            require_cross_fitted_review=False,
        )
        recommended = bool(
            support >= options.minimum_support
            and smoothed_emit - smoothed_abstain > options.abstention_margin
            and cross_validation["selected_fold_count"] > 0
            and cross_validation["mean_selected_delta"] > 0.0
            and cross_validation["minimum_selected_delta"]
            >= -options.maximum_fold_regression
        )
        groups.append(
            {
                **dict(
                    zip(
                        ("assertion", "entity_type", "scope", "rule_id", "cue"),
                        group_key,
                        strict=True,
                    )
                ),
                "support": support,
                "document_support": len({item.document_id for item in values}),
                "correct": correct,
                "raw_precision": round(correct / support, 6),
                "correct_probability_lower_95": round(
                    _wilson_lower(correct, support), 6
                ),
                "mean_emitted_jaccard": round(emitted_total / support, 6),
                "mean_abstained_jaccard": round(abstained_total / support, 6),
                "estimated_jaccard_gain": round(
                    smoothed_emit - smoothed_abstain, 6
                ),
                "cross_validation": cross_validation,
                "recommended": recommended,
            }
        )
    return {
        "schema_version": "phase1-assertion-calibration.v1",
        "options": {
            "folds": options.folds,
            "minimum_support": options.minimum_support,
            "minimum_train_support": options.minimum_train_support,
            "abstention_margin": options.abstention_margin,
            "maximum_fold_regression": options.maximum_fold_regression,
        },
        "coverage": {
            "gold_document_count": len(gold_by_document),
            "prediction_document_count": len(prediction_by_document),
            "observation_count": len(observations),
            "counters": dict(sorted(counters.items())),
        },
        "evidence_groups": groups,
        "recommended_rule_ids": sorted(
            {str(group["rule_id"]) for group in groups if group["recommended"]}
        ),
    }
def write_candidate_calibration_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidate_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = report.get("policy_groups", [])
    with (output / "candidate_calibration.jsonl").open("w", encoding="utf-8") as handle:
        if isinstance(rows, list):
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_assertion_calibration_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "assertion_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = report.get("evidence_groups", [])
    with (output / "assertion_calibration.jsonl").open("w", encoding="utf-8") as handle:
        if isinstance(rows, list):
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_calibrated_assertion_map(
    report: Mapping[str, Any],
    path: str | Path,
) -> list[dict[str, Any]]:
    raw_groups = report.get("evidence_groups", [])
    groups = raw_groups if isinstance(raw_groups, list) else []
    rows = [
        {
            "rule_id": str(group["rule_id"]),
            "assertion": str(group["assertion"]),
            "entity_type": str(group["entity_type"]),
            "scope": str(group["scope"]),
            "cue": str(group["cue"]),
            "support": int(group["support"]),
            "document_support": int(group["document_support"]),
            "raw_precision": float(group["raw_precision"]),
            "estimated_jaccard_gain": float(group["estimated_jaccard_gain"]),
            "review_status": "calibrated",
        }
        for group in groups
        if group.get("recommended") is True
    ]
    rows.sort(
        key=lambda row: (
            row["assertion"],
            row["entity_type"],
            row["rule_id"],
        )
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    return rows


def _group_report(
    observations: Sequence[_CandidateObservation],
    *,
    key: Any,
    key_fields: tuple[str, ...],
    options: CandidateCalibrationOptions,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[_CandidateObservation]] = defaultdict(list)
    for observation in observations:
        raw_key = key(observation)
        grouped[tuple(raw_key)].append(observation)
    rows: list[dict[str, Any]] = []
    for group_key, values in sorted(grouped.items(), key=lambda item: item[0]):
        statistics = _statistics(values, options)
        cross_validation = _cross_validate(
            values,
            options,
            require_cross_fitted_review=all(item.reviewed for item in values),
        )
        recommended = bool(
            len(values) >= options.minimum_support
            and statistics["estimated_jaccard_gain"] > options.abstention_margin
            and cross_validation["selected_fold_count"] > 0
            and cross_validation["mean_selected_delta"] > 0.0
            and cross_validation["minimum_selected_delta"]
            >= -options.maximum_fold_regression
        )
        rows.append(
            {
                **dict(zip(key_fields, group_key, strict=True)),
                **statistics,
                "cross_validation": cross_validation,
                "recommended": recommended,
            }
        )
    return rows


def _statistics(
    values: Sequence[_CandidateObservation],
    options: CandidateCalibrationOptions,
) -> dict[str, Any]:
    support = len(values)
    correct = sum(item.correct for item in values)
    emitted_total = sum(item.emitted_score for item in values)
    abstained_total = sum(item.abstained_score for item in values)
    smoothed_correct = _smoothed(correct, support, options)
    smoothed_emit = _smoothed(emitted_total, support, options)
    smoothed_abstain = _smoothed(abstained_total, support, options)
    return {
        "support": support,
        "document_support": len({item.document_id for item in values}),
        "correct": correct,
        "raw_precision": round(correct / support, 6),
        "smoothed_correct_probability": round(smoothed_correct, 6),
        "correct_probability_lower_95": round(_wilson_lower(correct, support), 6),
        "mean_retrieval_score": round(
            sum(item.retrieval_score for item in values) / support, 6
        ),
        "mean_emitted_jaccard": round(emitted_total / support, 6),
        "mean_abstained_jaccard": round(abstained_total / support, 6),
        "smoothed_emitted_jaccard": round(smoothed_emit, 6),
        "smoothed_abstained_jaccard": round(smoothed_abstain, 6),
        "estimated_jaccard_gain": round(smoothed_emit - smoothed_abstain, 6),
        "evidence_sources": sorted(
            {source for item in values for source in item.evidence_sources}
        ),
    }


def _cross_validate(
    values: Sequence[_CandidateObservation | _AssertionObservation],
    options: CandidateCalibrationOptions,
    *,
    require_cross_fitted_review: bool,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    selected_deltas: list[float] = []
    for fold in range(options.folds):
        eligible = (
            [
                item
                for item in values
                if isinstance(item, _CandidateObservation)
                and fold in item.reviewed_folds
            ]
            if require_cross_fitted_review
            else list(values)
        )
        train = [item for item in eligible if item.fold != fold]
        held_out = [item for item in eligible if item.fold == fold]
        if not held_out:
            continue
        train_emit = _smoothed(
            sum(item.emitted_score for item in train), len(train), options
        )
        train_abstain = _smoothed(
            sum(item.abstained_score for item in train), len(train), options
        )
        selected = bool(
            len(train) >= options.minimum_train_support
            and train_emit > train_abstain + options.abstention_margin
        )
        actual_delta = (
            sum(item.emitted_score - item.abstained_score for item in held_out)
            / len(held_out)
            if selected
            else 0.0
        )
        if selected:
            selected_deltas.append(actual_delta)
        fold_rows.append(
            {
                "fold": fold,
                "train_support": len(train),
                "held_out_support": len(held_out),
                "train_smoothed_emitted_jaccard": round(train_emit, 6),
                "train_smoothed_abstained_jaccard": round(train_abstain, 6),
                "selected": selected,
                "held_out_delta": round(actual_delta, 6),
            }
        )
    return {
        "folds": fold_rows,
        "selected_fold_count": len(selected_deltas),
        "mean_selected_delta": round(
            sum(selected_deltas) / len(selected_deltas), 6
        )
        if selected_deltas
        else 0.0,
        "minimum_selected_delta": round(min(selected_deltas), 6)
        if selected_deltas
        else 0.0,
        "cross_fitted_review_required": require_cross_fitted_review,
    }


def _top_candidate(entity: EntityAnnotation, expected_system: CodeSystem | None) -> Any:
    if expected_system is None:
        return None
    candidates = [
        candidate
        for candidate in entity.candidates
        if candidate.qualified
        and candidate.code_system == expected_system
        and candidate.code is not None
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda candidate: (-candidate.retrieval_score, str(candidate.code)),
    )[0]


def _gold_index(
    rows: Sequence[Mapping[str, Any]],
    counters: Counter[str],
) -> dict[tuple[int, int, str], Mapping[str, Any]]:
    output: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        position = row.get("position")
        entity_type = str(row.get("type", ""))
        if (
            not isinstance(position, list)
            or len(position) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in position)
        ):
            counters["invalid_gold_position"] += 1
            continue
        key = (int(position[0]), int(position[1]), entity_type)
        if key in output:
            counters["duplicate_gold_identity"] += 1
            continue
        output[key] = row
    return output


def _candidate_set(value: Any) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    return frozenset(str(item) for item in value if str(item))


def _cross_fitted_reviewed_candidates(
    gold_by_document: Mapping[str, list[dict[str, Any]]],
    reviewed_candidates: frozenset[tuple[str, str, str]],
    *,
    folds: int,
) -> dict[int, frozenset[tuple[str, str, str]]]:
    output: dict[int, frozenset[tuple[str, str, str]]] = {}
    for held_fold in range(folds):
        codes_by_mention: dict[tuple[str, str], set[str]] = defaultdict(set)
        for document_id, rows in gold_by_document.items():
            if _fold(document_id, folds) == held_fold:
                continue
            for row in rows:
                entity_type = str(row.get("type", ""))
                if entity_type not in PHASE1_CODABLE_TYPES:
                    continue
                codes = _candidate_set(row.get("candidates"))
                if len(codes) != 1:
                    continue
                mention = normalize_for_match(str(row.get("text", "")))
                if mention:
                    codes_by_mention[(mention, entity_type)].update(codes)
        candidates = {
            (mention, entity_type, next(iter(codes)))
            for (mention, entity_type), codes in codes_by_mention.items()
            if len(codes) == 1
        }
        output[held_fold] = frozenset(candidates & reviewed_candidates)
    return output


def _mention_structure(entity: EntityAnnotation, code_system: CodeSystem) -> str:
    if code_system != CodeSystem.RXNORM:
        return "not_applicable"
    medication = entity.medication_mention
    if medication is None:
        return "bare"
    kinds = {component.kind for component in medication.components}
    if kinds & {"strength", "dose_form", "dosage"}:
        return "structured"
    return "bare"


def _smoothed(
    successes: float,
    support: int,
    options: CandidateCalibrationOptions,
) -> float:
    return (successes + options.beta_alpha) / (
        support + options.beta_alpha + options.beta_beta
    )


def _wilson_lower(successes: int, support: int, *, z: float = 1.96) -> float:
    if support == 0:
        return 0.0
    probability = successes / support
    denominator = 1.0 + z * z / support
    center = probability + z * z / (2.0 * support)
    margin = z * math.sqrt(
        probability * (1.0 - probability) / support + z * z / (4.0 * support * support)
    )
    return max(0.0, (center - margin) / denominator)


def _fold(document_id: str, folds: int) -> int:
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % folds


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
