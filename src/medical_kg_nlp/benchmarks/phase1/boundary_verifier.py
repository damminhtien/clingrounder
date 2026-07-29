"""Train and apply a proposal-conditioned Phase 1 boundary ranker."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.boundary_variants import (
    PHASE1_BOUNDARY_FEATURE_CONTRACT,
    BoundaryErrorLabel,
    Phase1BoundaryVariant,
    boundary_cross_encoder_text,
    extract_phase1_boundary_features,
    generate_phase1_boundary_variants,
    label_phase1_boundary_variant,
)
from medical_kg_nlp.benchmarks.phase1.proposal_calibration import (
    Phase1ProposalVerifier,
    score_phase1_proposal_rows,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import (
    Phase1GenreBucket,
    ProposalSourceRole,
    extract_phase1_proposal_context,
    extract_phase1_proposal_features,
    is_phase1_heading_only_proposal,
    phase1_genre_bucket,
)
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from medical_kg_nlp.benchmarks.phase1.split_contract import (
    phase1_document_sort_key,
)
from medical_kg_nlp.evaluation.sparse_logistic import (
    SparseBinaryExample,
    SparseLogisticModel,
    SparseLogisticTrainingConfig,
    binary_probability_metrics,
    fit_sparse_logistic,
)
from medical_kg_nlp.ner.dictionary_matcher import DictionaryMatcher
from medical_kg_nlp.ner.document_structure import (
    DocumentStructure,
    DocumentStructureAnalyzer,
)
from medical_kg_nlp.ontology.phase1 import (
    PHASE1_ALLOWED_TYPES,
    PHASE1_CODABLE_TYPES,
    PHASE1_TYPE_PRIORITY,
)
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.io import read_jsonl

__all__ = [
    "Phase1BoundaryDataset",
    "Phase1BoundaryExample",
    "Phase1BoundaryVerifier",
    "ScoredPhase1BoundaryVariant",
    "build_phase1_boundary_dataset",
    "fit_phase1_boundary_verifier",
    "load_phase1_boundary_dataset",
    "resolve_phase1_boundary_rows",
    "write_phase1_boundary_dataset",
    "write_phase1_boundary_resolution",
    "write_phase1_boundary_verifier",
]

_DATASET_SCHEMA = "phase1-boundary-dataset.v1"
_VERIFIER_SCHEMA = "phase1-boundary-verifier.v2"
_REPORT_SCHEMA = "phase1-boundary-training-report.v2"
_RESOLUTION_SCHEMA = "phase1-boundary-resolution.v2"
_MIN_GENRE_PROPOSALS = 20
_MIN_GENRE_POSITIVES = 3
_MIN_GENRE_NEGATIVES = 3
_MAX_NEGATIVES_PER_POSITIVE_FAMILY = 4
_MAX_NEGATIVES_PER_UNCOVERED_FAMILY = 2


@dataclass(frozen=True, slots=True)
class Phase1BoundaryExample:
    """One candidate variant with its exact-boundary target and joint input."""

    variant: Phase1BoundaryVariant
    split: str
    label: int
    error_label: str
    genre: str
    section: str
    cross_encoder_text: str
    features: tuple[tuple[str, float], ...]
    base_probability: float | None
    base_selected: bool

    def __post_init__(self) -> None:
        if self.split not in {"train", "development"}:
            raise ValueError("Boundary example split must be train or development")
        if self.label not in {0, 1}:
            raise ValueError("Boundary example label must be binary")
        error = BoundaryErrorLabel(self.error_label)
        if (self.label == 1) != (error is BoundaryErrorLabel.CORRECT):
            raise ValueError("Only CORRECT boundary variants may be positive")
        Phase1GenreBucket(self.genre)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.variant.to_dict(),
            "split": self.split,
            "label": self.label,
            "error_label": self.error_label,
            "genre": self.genre,
            "section": self.section,
            "cross_encoder_text": self.cross_encoder_text,
            "features": dict(self.features),
            "base_probability": self.base_probability,
            "base_selected": self.base_selected,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Phase1BoundaryExample:
        """Restore one materialized row without regenerating boundary candidates."""

        raw_features = payload.get("features")
        if not isinstance(raw_features, Mapping):
            raise ValueError("Boundary example features must be an object")
        features: list[tuple[str, float]] = []
        for name, raw_value in raw_features.items():
            if (
                not isinstance(name, str)
                or not isinstance(raw_value, int | float)
                or isinstance(raw_value, bool)
            ):
                raise ValueError("Boundary example feature is malformed")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("Boundary example feature must be finite")
            features.append((name, value))
        raw_probability = payload.get("base_probability")
        base_probability = (
            None
            if raw_probability is None
            else _numeric_threshold(raw_probability)
        )
        return cls(
            variant=Phase1BoundaryVariant.from_dict(payload),
            split=str(payload.get("split", "")),
            label=_binary_value(payload.get("label"), "label"),
            error_label=str(payload.get("error_label", "")),
            genre=str(payload.get("genre", "")),
            section=str(payload.get("section", "")),
            cross_encoder_text=str(payload.get("cross_encoder_text", "")),
            features=tuple(sorted(features)),
            base_probability=base_probability,
            base_selected=_boolean_value(
                payload.get("base_selected"),
                "base_selected",
            ),
        )


@dataclass(frozen=True, slots=True)
class Phase1BoundaryDataset:
    """Leakage-safe boundary examples plus their deterministic manifest."""

    examples: tuple[Phase1BoundaryExample, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ScoredPhase1BoundaryVariant:
    """Runtime rank decision for one boundary option."""

    variant: Phase1BoundaryVariant
    genre: str
    probability: float
    threshold: float
    family_winner: bool
    selected_before_overlap: bool
    selected: bool
    rejection_reason: str | None
    resolution_policy: str
    base_selected: bool
    replacement_selected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant.to_dict(),
            "genre": self.genre,
            "probability": self.probability,
            "threshold": self.threshold,
            "family_winner": self.family_winner,
            "selected_before_overlap": self.selected_before_overlap,
            "selected": self.selected,
            "rejection_reason": self.rejection_reason,
            "resolution_policy": self.resolution_policy,
            "base_selected": self.base_selected,
            "replacement_selected": self.replacement_selected,
        }


@dataclass(frozen=True, slots=True)
class Phase1BoundaryVerifier:
    """Shared sparse boundary scorer with genre/type development operating points."""

    model: SparseLogisticModel
    thresholds: tuple[tuple[str, float], ...]
    genre_thresholds: tuple[tuple[str, str, float], ...]
    replacement_margins: tuple[tuple[str, float], ...]
    resolution_policy: str
    training_dataset_sha256: str
    requires_base_probability: bool

    def __post_init__(self) -> None:
        threshold_types = {entity_type for entity_type, _ in self.thresholds}
        if threshold_types != set(PHASE1_ALLOWED_TYPES):
            raise ValueError("Boundary verifier must define all Phase 1 thresholds")
        if len(self.thresholds) != len(threshold_types):
            raise ValueError("Boundary verifier contains duplicate global thresholds")
        if any(not 0.0 <= value <= 1.0 for _, value in self.thresholds):
            raise ValueError("Boundary verifier thresholds must be within [0, 1]")
        keys: set[tuple[str, str]] = set()
        for genre, entity_type, threshold in self.genre_thresholds:
            Phase1GenreBucket(genre)
            if entity_type not in PHASE1_ALLOWED_TYPES:
                raise ValueError("Boundary verifier genre threshold type is invalid")
            if (genre, entity_type) in keys:
                raise ValueError("Boundary verifier has duplicate genre thresholds")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("Boundary verifier genre threshold is invalid")
            keys.add((genre, entity_type))
        replacement_types = {
            entity_type for entity_type, _ in self.replacement_margins
        }
        if replacement_types != set(PHASE1_ALLOWED_TYPES):
            raise ValueError("Boundary verifier must define all replacement margins")
        if any(
            not 0.0 <= margin <= 2.0
            for _, margin in self.replacement_margins
        ):
            raise ValueError("Boundary replacement margins must be within [0, 2]")
        if self.resolution_policy not in {"open_ranker", "conservative_replacement"}:
            raise ValueError("Boundary verifier resolution policy is invalid")
        if (
            self.resolution_policy == "conservative_replacement"
            and not self.requires_base_probability
        ):
            raise ValueError(
                "Conservative boundary replacement requires a proposal verifier"
            )
        if len(self.training_dataset_sha256) != 64:
            raise ValueError("Boundary verifier requires a dataset SHA-256")

    @property
    def threshold_by_type(self) -> dict[str, float]:
        return dict(self.thresholds)

    @property
    def genre_threshold_by_key(self) -> dict[tuple[str, str], float]:
        return {
            (genre, entity_type): threshold
            for genre, entity_type, threshold in self.genre_thresholds
        }

    @property
    def replacement_margin_by_type(self) -> dict[str, float]:
        return dict(self.replacement_margins)

    def predict_probability(self, features: Mapping[str, float]) -> float:
        """Estimate P(exact raw span and exact type) for one variant."""

        if self.requires_base_probability and "numeric:base_proposal_probability" not in features:
            raise ValueError("Boundary verifier requires a frozen proposal probability")
        return self.model.predict_probability(features)

    def threshold_for(
        self,
        entity_type: str,
        *,
        genre: Phase1GenreBucket | str,
    ) -> float:
        bucket = Phase1GenreBucket(genre)
        return self.genre_threshold_by_key.get(
            (bucket.value, entity_type),
            self.threshold_by_type[entity_type],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _VERIFIER_SCHEMA,
            "feature_contract": PHASE1_BOUNDARY_FEATURE_CONTRACT,
            "training_dataset_sha256": self.training_dataset_sha256,
            "requires_base_probability": self.requires_base_probability,
            "resolution_policy": self.resolution_policy,
            "thresholds": dict(self.thresholds),
            "genre_thresholds": _nested_genre_thresholds(self.genre_thresholds),
            "replacement_margins": dict(self.replacement_margins),
            "model": self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Phase1BoundaryVerifier:
        if payload.get("schema_version") != _VERIFIER_SCHEMA:
            raise ValueError("Unsupported Phase 1 boundary verifier schema")
        if payload.get("feature_contract") != PHASE1_BOUNDARY_FEATURE_CONTRACT:
            raise ValueError("Boundary verifier feature contract is incompatible")
        raw_thresholds = payload.get("thresholds")
        raw_genre_thresholds = payload.get("genre_thresholds")
        raw_replacement_margins = payload.get("replacement_margins")
        raw_model = payload.get("model")
        if (
            not isinstance(raw_thresholds, Mapping)
            or not isinstance(raw_genre_thresholds, Mapping)
            or not isinstance(raw_replacement_margins, Mapping)
            or not isinstance(raw_model, Mapping)
        ):
            raise ValueError("Boundary verifier artifact is incomplete")
        thresholds = tuple(
            sorted(
                (str(entity_type), _numeric_threshold(value))
                for entity_type, value in raw_thresholds.items()
            )
        )
        return cls(
            model=SparseLogisticModel.from_dict(raw_model),
            thresholds=thresholds,
            genre_thresholds=_parse_genre_thresholds(raw_genre_thresholds),
            replacement_margins=tuple(
                sorted(
                    (str(entity_type), _numeric_margin(value))
                    for entity_type, value in raw_replacement_margins.items()
                )
            ),
            resolution_policy=str(payload.get("resolution_policy", "")),
            training_dataset_sha256=str(payload.get("training_dataset_sha256", "")),
            requires_base_probability=bool(
                payload.get("requires_base_probability", False)
            ),
        )


def build_phase1_boundary_dataset(
    matrix_path: str | Path,
    corpus: Phase1ReviewedCorpus,
    *,
    source_roles: Mapping[str, ProposalSourceRole | str],
    proposal_verifier: Phase1ProposalVerifier | None = None,
    proposal_verifier_path: str | Path | None = None,
    dictionary_matcher: DictionaryMatcher | None = None,
    corpus_fingerprint_sha256: str,
) -> Phase1BoundaryDataset:
    """Build candidate-level labels without reading any frozen holdout annotation."""

    matrix_file = Path(matrix_path)
    rows_by_document = _rows_by_document(read_jsonl(matrix_file))
    analyzer = DocumentStructureAnalyzer()
    examples: list[Phase1BoundaryExample] = []
    gold_counts: Counter[str] = Counter()
    genre_counts: Counter[str] = Counter()
    for document_id in sorted(
        corpus.source_texts,
        key=phase1_document_sort_key,
    ):
        source_text = corpus.source_texts[document_id]
        structure = analyzer.analyze(source_text)
        genre = phase1_genre_bucket(structure.genre)
        rows = rows_by_document.get(document_id, ())
        base_selected_keys = _base_selected_keys(
            rows,
            document_id,
            source_text,
            source_roles,
            proposal_verifier,
        )
        variants = generate_phase1_boundary_variants(
            document_id,
            source_text,
            rows,
            source_roles=source_roles,
            dictionary_matcher=dictionary_matcher,
            structure=structure,
        )
        family_sizes = Counter(variant.family_id for variant in variants)
        gold_rows = corpus.gold_rows[document_id]
        split = corpus.split_by_document[document_id]
        for row in gold_rows:
            entity_type = str(row.get("type", ""))
            gold_counts[f"{split}:{entity_type}"] += 1
            gold_counts[f"{split}:{genre.value}:{entity_type}"] += 1
        for variant in variants:
            base_probability = _base_probability(
                variant,
                source_text,
                structure,
                source_roles,
                proposal_verifier,
            )
            error = label_phase1_boundary_variant(variant, gold_rows)
            context = extract_phase1_proposal_context(
                variant.to_proposal_row(),
                source_text,
                structure=structure,
            )
            features = extract_phase1_boundary_features(
                variant,
                source_text,
                source_roles,
                family_size=family_sizes[variant.family_id],
                base_probability=base_probability,
                structure=structure,
            )
            examples.append(
                Phase1BoundaryExample(
                    variant=variant,
                    split=split,
                    label=int(error is BoundaryErrorLabel.CORRECT),
                    error_label=error.value,
                    genre=genre.value,
                    section=context.section,
                    cross_encoder_text=boundary_cross_encoder_text(
                        variant,
                        source_text,
                        structure=structure,
                    ),
                    features=tuple(sorted(features.items())),
                    base_probability=base_probability,
                    base_selected=(
                        variant.position,
                        variant.entity_type,
                    )
                    in base_selected_keys,
                )
            )
            genre_counts[f"{split}:{genre.value}"] += 1
    examples.sort(key=_example_sort_key)
    serialized = _serialize_examples(examples)
    manifest = {
        "schema_version": _DATASET_SCHEMA,
        "feature_contract": PHASE1_BOUNDARY_FEATURE_CONTRACT,
        "example_count": len(examples),
        "examples_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "corpus_fingerprint_sha256": corpus_fingerprint_sha256,
        "round2_included": False,
        "holdout_labels_read": False,
        "requires_base_probability": proposal_verifier is not None,
        "inputs": {
            "proposal_matrix_sha256": sha256_file(matrix_file),
            "proposal_verifier_sha256": (
                sha256_file(proposal_verifier_path)
                if proposal_verifier_path is not None
                else None
            ),
        },
        "source_roles": {
            source: ProposalSourceRole(role).value
            for source, role in sorted(source_roles.items())
        },
        "split_counts": dict(
            sorted(Counter(example.split for example in examples).items())
        ),
        "label_counts": dict(
            sorted(
                Counter(
                    f"{example.split}:{example.error_label}"
                    for example in examples
                ).items()
            )
        ),
        "generator_counts": dict(
            sorted(
                Counter(
                    f"{example.split}:{generator}"
                    for example in examples
                    for generator in example.variant.generators
                ).items()
            )
        ),
        "genre_counts": dict(sorted(genre_counts.items())),
        "gold_entity_counts": dict(sorted(gold_counts.items())),
    }
    return Phase1BoundaryDataset(tuple(examples), manifest)


def fit_phase1_boundary_verifier(
    dataset: Phase1BoundaryDataset,
    *,
    training_config: SparseLogisticTrainingConfig | None = None,
) -> tuple[Phase1BoundaryVerifier, dict[str, Any]]:
    """Fit the shared scorer and calibrate type/genre thresholds on development."""

    if dataset.manifest.get("feature_contract") != PHASE1_BOUNDARY_FEATURE_CONTRACT:
        raise ValueError("Boundary dataset feature contract is incompatible")
    train = tuple(example for example in dataset.examples if example.split == "train")
    development = tuple(
        example for example in dataset.examples if example.split == "development"
    )
    if not train or not development:
        raise ValueError("Boundary verifier requires train and development examples")
    sampled_train = _sample_training_examples(train)
    family_sizes = Counter(
        example.variant.family_id for example in sampled_train
    )
    positive_weight, negative_weight = _balanced_class_weights(
        sampled_train,
        family_sizes,
    )
    sparse_train = [
        SparseBinaryExample.from_mapping(
            dict(example.features),
            label=example.label,
            weight=(
                positive_weight if example.label else negative_weight
            )
            / family_sizes[example.variant.family_id],
        )
        for example in sampled_train
    ]
    model, training_report = fit_sparse_logistic(
        sparse_train,
        config=training_config
        or SparseLogisticTrainingConfig(
            epochs=100,
            learning_rate=0.3,
            learning_rate_decay=0.02,
            l2=0.002,
            tolerance=1e-8,
        ),
    )
    probabilities = tuple(
        model.predict_probability(dict(example.features))
        for example in development
    )
    gold_counts = _gold_counts(dataset.manifest)
    thresholds, threshold_report = _calibrate_thresholds(
        development,
        probabilities,
        gold_counts,
    )
    genre_thresholds, genre_threshold_report = _calibrate_genre_thresholds(
        development,
        probabilities,
        gold_counts,
    )
    selected = _select_examples(
        development,
        probabilities,
        thresholds,
        genre_thresholds=genre_thresholds,
    )
    dataset_sha256 = str(dataset.manifest.get("examples_sha256", ""))
    replacement_margins, replacement_report = _calibrate_replacement_margins(
        development,
        probabilities,
        gold_counts,
    )
    conservative_selected = _select_conservative_examples(
        development,
        probabilities,
        replacement_margins,
    )
    verifier = Phase1BoundaryVerifier(
        model=model,
        thresholds=tuple(sorted(thresholds.items())),
        genre_thresholds=tuple(
            sorted(
                (genre, entity_type, threshold)
                for (genre, entity_type), threshold in genre_thresholds.items()
            )
        ),
        replacement_margins=tuple(sorted(replacement_margins.items())),
        resolution_policy=(
            "conservative_replacement"
            if dataset.manifest.get("requires_base_probability", False)
            else "open_ranker"
        ),
        training_dataset_sha256=dataset_sha256,
        requires_base_probability=bool(
            dataset.manifest.get("requires_base_probability", False)
        ),
    )
    report = {
        "schema_version": _REPORT_SCHEMA,
        "feature_contract": PHASE1_BOUNDARY_FEATURE_CONTRACT,
        "training_dataset_sha256": dataset_sha256,
        "holdout_opened": False,
        "evaluation_scope": "diagnostic_only_manual_gold",
        "training": {
            **training_report,
            "family_weighting": True,
            "source_example_count": len(train),
            "sampled_example_count": len(sampled_train),
            "hard_negative_sampling": {
                "positive_family_limit": _MAX_NEGATIVES_PER_POSITIVE_FAMILY,
                "uncovered_family_limit": _MAX_NEGATIVES_PER_UNCOVERED_FAMILY,
            },
            "positive_class_weight": positive_weight,
            "negative_class_weight": negative_weight,
        },
        "development_probability_metrics": binary_probability_metrics(
            [example.label for example in development],
            probabilities,
        ),
        "threshold_calibration": threshold_report,
        "genre_threshold_calibration": genre_threshold_report,
        "replacement_margin_calibration": replacement_report,
        "development_selection": {
            "learned": _selection_metrics(
                selected,
                gold_counts,
                split="development",
            ),
            "base_proposal_verifier": _selection_metrics(
                [example for example in development if example.base_selected],
                gold_counts,
                split="development",
            ),
            "conservative_boundary_replacement": _selection_metrics(
                conservative_selected,
                gold_counts,
                split="development",
            ),
            "active": _selection_metrics(
                (
                    conservative_selected
                    if dataset.manifest.get("requires_base_probability", False)
                    else selected
                ),
                gold_counts,
                split="development",
            ),
            "family_top1": _family_top1_metrics(development, probabilities),
            "by_genre": _selection_by_genre(
                selected,
                gold_counts,
                split="development",
            ),
        },
        "candidate_coverage": {
            split: _coverage_metrics(dataset.examples, gold_counts, split)
            for split in ("train", "development")
        },
        "top_weights": _top_weights(model),
    }
    report["promotion_gate"] = _diagnostic_promotion_gate(report)
    return verifier, report


def resolve_phase1_boundary_rows(
    rows: Sequence[Mapping[str, Any]],
    source_text_by_document: Mapping[str, str],
    verifier: Phase1BoundaryVerifier,
    *,
    source_roles: Mapping[str, ProposalSourceRole | str],
    proposal_verifier: Phase1ProposalVerifier | None = None,
    dictionary_matcher: DictionaryMatcher | None = None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    tuple[ScoredPhase1BoundaryVariant, ...],
]:
    """Rank boundary families, abstain by genre/type, then resolve cross-type overlap."""

    if verifier.requires_base_probability != (proposal_verifier is not None):
        raise ValueError(
            "Boundary runtime must match the training base-probability contract"
        )
    rows_by_document = _rows_by_document(rows)
    analyzer = DocumentStructureAnalyzer()
    provisional: list[ScoredPhase1BoundaryVariant] = []
    conservative_accepted: set[str] = set()
    for document_id in sorted(
        source_text_by_document,
        key=phase1_document_sort_key,
    ):
        source_text = source_text_by_document[document_id]
        structure = analyzer.analyze(source_text)
        genre = phase1_genre_bucket(structure.genre)
        variants = generate_phase1_boundary_variants(
            document_id,
            source_text,
            rows_by_document.get(document_id, ()),
            source_roles=source_roles,
            dictionary_matcher=dictionary_matcher,
            structure=structure,
        )
        family_sizes = Counter(variant.family_id for variant in variants)
        scored_variants = [
            (
                variant,
                verifier.predict_probability(
                    extract_phase1_boundary_features(
                        variant,
                        source_text,
                        source_roles,
                        family_size=family_sizes[variant.family_id],
                        base_probability=_base_probability(
                            variant,
                            source_text,
                            structure,
                            source_roles,
                            proposal_verifier,
                        ),
                        structure=structure,
                    )
                ),
            )
            for variant in variants
        ]
        winners = {
            winner.variant_id
            for winner, _ in _family_winners(scored_variants)
        }
        base_selected_keys = _base_selected_keys(
            rows_by_document.get(document_id, ()),
            document_id,
            source_text,
            source_roles,
            proposal_verifier,
        )
        structural_heading_ids = {
            variant.variant_id
            for variant, _ in scored_variants
            if is_phase1_heading_only_proposal(
                variant.to_proposal_row(),
                source_text,
                structure=structure,
            )
        }
        replacement_ids: set[str] = set()
        if verifier.resolution_policy == "conservative_replacement":
            document_selected, replacement_ids = _conservative_runtime_selection(
                scored_variants,
                base_selected_keys,
                verifier.replacement_margin_by_type,
                excluded_variant_ids=structural_heading_ids,
            )
            conservative_accepted.update(document_selected)
        for variant, probability in scored_variants:
            family_winner = variant.variant_id in winners
            base_selected = (
                variant.position,
                variant.entity_type,
            ) in base_selected_keys
            threshold = (
                verifier.replacement_margin_by_type[variant.entity_type]
                if verifier.resolution_policy == "conservative_replacement"
                else verifier.threshold_for(
                    variant.entity_type,
                    genre=genre,
                )
            )
            structural_heading = variant.variant_id in structural_heading_ids
            selected_before_overlap = (
                variant.variant_id in conservative_accepted
                if verifier.resolution_policy == "conservative_replacement"
                else (
                    family_winner
                    and probability >= threshold
                    and not structural_heading
                )
            )
            if verifier.resolution_policy == "conservative_replacement":
                rejection_reason = (
                    None
                    if selected_before_overlap
                    else (
                        "structural_heading"
                        if structural_heading
                        else "not_selected_by_conservative_replacement"
                    )
                )
            else:
                rejection_reason = (
                    "family_lower_rank"
                    if not family_winner
                    else (
                        "structural_heading"
                        if structural_heading
                        else (
                            "below_threshold"
                            if probability < threshold
                            else "overlap_pending"
                        )
                    )
                )
            provisional.append(
                ScoredPhase1BoundaryVariant(
                    variant=variant,
                    genre=genre.value,
                    probability=probability,
                    threshold=threshold,
                    family_winner=family_winner,
                    selected_before_overlap=selected_before_overlap,
                    selected=False,
                    rejection_reason=rejection_reason,
                    resolution_policy=verifier.resolution_policy,
                    base_selected=base_selected,
                    replacement_selected=(
                        variant.variant_id in replacement_ids
                    ),
                )
            )
    accepted = (
        conservative_accepted
        if verifier.resolution_policy == "conservative_replacement"
        else {
            item.variant.variant_id
            for item in _resolve_scored_overlaps(
                [item for item in provisional if item.selected_before_overlap]
            )
        }
    )
    scored = tuple(
        ScoredPhase1BoundaryVariant(
            variant=item.variant,
            genre=item.genre,
            probability=item.probability,
            threshold=item.threshold,
            family_winner=item.family_winner,
            selected_before_overlap=item.selected_before_overlap,
            selected=item.variant.variant_id in accepted,
            rejection_reason=(
                None
                if item.variant.variant_id in accepted
                else (
                    "overlap"
                    if item.selected_before_overlap
                    else item.rejection_reason
                )
            ),
            resolution_policy=item.resolution_policy,
            base_selected=item.base_selected,
            replacement_selected=item.replacement_selected,
        )
        for item in provisional
    )
    output: dict[str, list[dict[str, Any]]] = {
        document_id: [] for document_id in source_text_by_document
    }
    for item in scored:
        if not item.selected:
            continue
        variant = item.variant
        start, end = variant.position
        if source_text_by_document[variant.document_id][start:end] != variant.text:
            raise ValueError("Selected boundary variant violates raw offset invariant")
        entity: dict[str, Any] = {
            "text": variant.text,
            "type": variant.entity_type,
            "assertions": [],
            "position": [start, end],
        }
        if variant.entity_type in PHASE1_CODABLE_TYPES:
            entity["candidates"] = []
        output[variant.document_id].append(entity)
    for values in output.values():
        values.sort(key=_output_sort_key)
    return output, scored


def write_phase1_boundary_dataset(
    dataset: Phase1BoundaryDataset,
    output_dir: str | Path,
) -> None:
    """Write inspectable sparse and cross-encoder training rows."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(dataset.manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output / "examples.jsonl").write_text(
        _serialize_examples(dataset.examples),
        encoding="utf-8",
    )


def load_phase1_boundary_dataset(
    input_dir: str | Path,
) -> Phase1BoundaryDataset:
    """Load a materialized dataset and reject partial or modified artifacts.

    SCALING: candidate generation can produce hundreds of megabytes. Loading the immutable
    artifact makes an interrupted training run resumable without repeating that work.
    """

    source = Path(input_dir)
    manifest_path = source / "manifest.json"
    examples_path = source / "examples.jsonl"
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, Mapping):
        raise ValueError("Boundary dataset manifest must be an object")
    manifest = dict(raw_manifest)
    if manifest.get("schema_version") != _DATASET_SCHEMA:
        raise ValueError("Unsupported Phase 1 boundary dataset schema")
    expected_sha256 = manifest.get("examples_sha256")
    if (
        not isinstance(expected_sha256, str)
        or sha256_file(examples_path) != expected_sha256
    ):
        raise ValueError("Boundary dataset examples SHA-256 does not match manifest")
    examples: list[Phase1BoundaryExample] = []
    # SCALING: stream rows so the large JSONL is not duplicated as a second list of dictionaries.
    with examples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"Boundary dataset row {line_number} must be an object"
                )
            examples.append(Phase1BoundaryExample.from_dict(payload))
    if len(examples) != manifest.get("example_count"):
        raise ValueError("Boundary dataset example count does not match manifest")
    return Phase1BoundaryDataset(tuple(examples), manifest)


def write_phase1_boundary_verifier(
    verifier: Phase1BoundaryVerifier,
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "verifier.json").write_text(
        json.dumps(verifier.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_phase1_boundary_resolution(
    rows_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    scored: Sequence[ScoredPhase1BoundaryVariant],
    output_dir: str | Path,
    *,
    matrix_path: str | Path,
    verifier_path: str | Path,
    proposal_verifier_path: str | Path | None,
    dictionary_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Write deterministic entity files, per-variant trace, and provenance."""

    output = Path(output_dir)
    entity_dir = output / "output"
    entity_dir.mkdir(parents=True, exist_ok=True)
    for document_id in sorted(rows_by_document, key=phase1_document_sort_key):
        (entity_dir / f"{document_id}.json").write_text(
            json.dumps(
                list(rows_by_document[document_id]),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    (output / "boundary_scores.jsonl").write_text(
        "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in scored
        ),
        encoding="utf-8",
    )
    rejection_counts = Counter(
        item.rejection_reason or "selected" for item in scored
    )
    manifest = {
        "schema_version": _RESOLUTION_SCHEMA,
        "feature_contract": PHASE1_BOUNDARY_FEATURE_CONTRACT,
        "holdout_labels_opened": False,
        "inputs": {
            "matrix_sha256": sha256_file(matrix_path),
            "boundary_verifier_sha256": sha256_file(verifier_path),
            "proposal_verifier_sha256": (
                sha256_file(proposal_verifier_path)
                if proposal_verifier_path is not None
                else None
            ),
            "dictionaries": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path in dictionary_paths
            ],
        },
        "document_count": len(rows_by_document),
        "variant_count": len(scored),
        "family_count": len({item.variant.family_id for item in scored}),
        "selected_count": sum(item.selected for item in scored),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _base_probability(
    variant: Phase1BoundaryVariant,
    source_text: str,
    structure: DocumentStructure,
    source_roles: Mapping[str, ProposalSourceRole | str],
    proposal_verifier: Phase1ProposalVerifier | None,
) -> float | None:
    if proposal_verifier is None:
        return None
    features = extract_phase1_proposal_features(
        variant.to_proposal_row(),
        source_text,
        source_roles,
        structure=structure,
    )
    return proposal_verifier.predict_probability(
        features,
        genre=phase1_genre_bucket(structure.genre),
    )


def _base_selected_keys(
    rows: Sequence[Mapping[str, Any]],
    document_id: str,
    source_text: str,
    source_roles: Mapping[str, ProposalSourceRole | str],
    proposal_verifier: Phase1ProposalVerifier | None,
) -> set[tuple[tuple[int, int], str]]:
    if proposal_verifier is None:
        return set()
    scored = score_phase1_proposal_rows(
        rows,
        {document_id: source_text},
        proposal_verifier,
        source_roles=source_roles,
    )
    return {
        (
            _position(item.row),
            str(item.row.get("type", "")),
        )
        for item in scored
        if item.selected
    }


def _balanced_class_weights(
    examples: Sequence[Phase1BoundaryExample],
    family_sizes: Mapping[str, int],
) -> tuple[float, float]:
    positive = sum(
        1.0 / family_sizes[example.variant.family_id]
        for example in examples
        if example.label
    )
    negative = sum(
        1.0 / family_sizes[example.variant.family_id]
        for example in examples
        if not example.label
    )
    if positive <= 0.0 or negative <= 0.0:
        raise ValueError("Boundary training requires both exact and incorrect variants")
    total = positive + negative
    return total / (2.0 * positive), total / (2.0 * negative)


def _sample_training_examples(
    examples: Sequence[Phase1BoundaryExample],
) -> tuple[Phase1BoundaryExample, ...]:
    """Keep exact variants and a bounded, diverse set of hard negatives per family."""

    grouped: dict[str, list[Phase1BoundaryExample]] = defaultdict(list)
    for example in examples:
        grouped[example.variant.family_id].append(example)
    sampled: list[Phase1BoundaryExample] = []
    for family_id in sorted(grouped):
        values = grouped[family_id]
        positives = [example for example in values if example.label]
        negatives = sorted(
            (example for example in values if not example.label),
            key=_hard_negative_key,
        )
        sampled.extend(positives)
        limit = (
            _MAX_NEGATIVES_PER_POSITIVE_FAMILY
            if positives
            else _MAX_NEGATIVES_PER_UNCOVERED_FAMILY
        )
        selected_negatives: list[Phase1BoundaryExample] = []
        selected_ids: set[str] = set()
        # Boundary labels are distinct failure modes; retain one of each before filling by score.
        for error_label in (
            BoundaryErrorLabel.TOO_SHORT.value,
            BoundaryErrorLabel.TOO_LONG.value,
            BoundaryErrorLabel.WRONG_ENTITY.value,
        ):
            candidate = next(
                (
                    example
                    for example in negatives
                    if example.error_label == error_label
                    and example.variant.variant_id not in selected_ids
                ),
                None,
            )
            if candidate is not None and len(selected_negatives) < limit:
                selected_negatives.append(candidate)
                selected_ids.add(candidate.variant.variant_id)
        for example in negatives:
            if len(selected_negatives) >= limit:
                break
            if example.variant.variant_id not in selected_ids:
                selected_negatives.append(example)
                selected_ids.add(example.variant.variant_id)
        sampled.extend(selected_negatives)
    return tuple(sorted(sampled, key=_example_sort_key))


def _hard_negative_key(example: Phase1BoundaryExample) -> tuple[Any, ...]:
    variant = example.variant
    return (
        -(example.base_probability or 0.0),
        -int(variant.position in variant.foundation_spans),
        -len(variant.sources),
        0
        if example.error_label
        in {
            BoundaryErrorLabel.TOO_SHORT.value,
            BoundaryErrorLabel.TOO_LONG.value,
        }
        else 1,
        variant.variant_id,
    )


def _calibrate_thresholds(
    examples: Sequence[Phase1BoundaryExample],
    probabilities: Sequence[float],
    gold_counts: Mapping[tuple[str, str, str | None], int],
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds: dict[str, float] = {}
    report: dict[str, Any] = {}
    for entity_type in sorted(PHASE1_ALLOWED_TYPES):
        indices = [
            index
            for index, example in enumerate(examples)
            if example.variant.entity_type == entity_type
        ]
        scoped = tuple(examples[index] for index in indices)
        scoped_probabilities = tuple(probabilities[index] for index in indices)
        gold = gold_counts.get(("development", entity_type, None), 0)
        threshold, metrics, candidate_count = _best_threshold(
            scoped,
            scoped_probabilities,
            gold,
        )
        thresholds[entity_type] = threshold
        report[entity_type] = {
            "proposal_support": len(scoped),
            "positive_support": sum(example.label for example in scoped),
            "gold_support": gold,
            "candidate_threshold_count": candidate_count,
            "threshold": threshold,
            "metrics": metrics,
        }
    return thresholds, report


def _calibrate_genre_thresholds(
    examples: Sequence[Phase1BoundaryExample],
    probabilities: Sequence[float],
    gold_counts: Mapping[tuple[str, str, str | None], int],
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    thresholds: dict[tuple[str, str], float] = {}
    report: dict[str, Any] = {}
    for genre in (
        Phase1GenreBucket.CLINICAL,
        Phase1GenreBucket.QUESTION_ANSWER,
        Phase1GenreBucket.EDUCATIONAL,
    ):
        genre_report: dict[str, Any] = {}
        for entity_type in sorted(PHASE1_ALLOWED_TYPES):
            indices = [
                index
                for index, example in enumerate(examples)
                if example.genre == genre.value
                and example.variant.entity_type == entity_type
            ]
            scoped = tuple(examples[index] for index in indices)
            scoped_probabilities = tuple(probabilities[index] for index in indices)
            positive = sum(example.label for example in scoped)
            negative = len(scoped) - positive
            gold = gold_counts.get(
                ("development", entity_type, genre.value),
                0,
            )
            values: dict[str, Any] = {
                "proposal_support": len(scoped),
                "positive_support": positive,
                "negative_support": negative,
                "gold_support": gold,
                "selected": False,
            }
            if (
                len(scoped) < _MIN_GENRE_PROPOSALS
                or positive < _MIN_GENRE_POSITIVES
                or negative < _MIN_GENRE_NEGATIVES
                or gold < _MIN_GENRE_POSITIVES
            ):
                values["fallback_reason"] = "insufficient_development_support"
                genre_report[entity_type] = values
                continue
            threshold, metrics, candidate_count = _best_threshold(
                scoped,
                scoped_probabilities,
                gold,
            )
            thresholds[(genre.value, entity_type)] = threshold
            values.update(
                {
                    "selected": True,
                    "fallback_reason": None,
                    "threshold": threshold,
                    "candidate_threshold_count": candidate_count,
                    "metrics": metrics,
                }
            )
            genre_report[entity_type] = values
        report[genre.value] = genre_report
    return thresholds, report


def _best_threshold(
    examples: Sequence[Phase1BoundaryExample],
    probabilities: Sequence[float],
    gold: int,
) -> tuple[float, dict[str, Any], int]:
    winners = _family_winners(
        [
            (example.variant, probability)
            for example, probability in zip(examples, probabilities, strict=True)
        ]
    )
    example_by_id = {
        example.variant.variant_id: example for example in examples
    }
    ranked = sorted(
        (
            (example_by_id[variant.variant_id], probability)
            for variant, probability in winners
        ),
        key=lambda item: (-item[1], item[0].variant.variant_id),
    )
    candidates = {0.0, 1.0, *(probability for _, probability in ranked)}
    best_rank = _metric_rank(0, 0, gold, 1.0)
    best_threshold = 1.0
    true_positive = 0
    false_positive = 0
    index = 0
    while index < len(ranked):
        probability = ranked[index][1]
        while index < len(ranked) and ranked[index][1] == probability:
            example = ranked[index][0]
            true_positive += example.label
            false_positive += 1 - example.label
            index += 1
        rank = _metric_rank(
            true_positive,
            false_positive,
            gold,
            probability,
        )
        if rank > best_rank:
            best_rank = rank
            best_threshold = probability
    all_rank = _metric_rank(true_positive, false_positive, gold, 0.0)
    if all_rank > best_rank:
        best_threshold = 0.0
    selected = tuple(
        example for example, probability in ranked if probability >= best_threshold
    )
    return (
        best_threshold,
        _metrics_against_gold(selected, gold),
        len(candidates),
    )


def _metric_rank(
    true_positive: int,
    false_positive: int,
    gold: int,
    threshold: float,
) -> tuple[float, float, float, float]:
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / gold if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall, threshold


def _calibrate_replacement_margins(
    examples: Sequence[Phase1BoundaryExample],
    probabilities: Sequence[float],
    gold_counts: Mapping[tuple[str, str, str | None], int],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Tune per-type correction margins while retaining the frozen proposal set."""

    margins = {entity_type: 2.0 for entity_type in PHASE1_ALLOWED_TYPES}
    gold = sum(
        count
        for (split, _, genre), count in gold_counts.items()
        if split == "development" and genre is None
    )
    report: dict[str, Any] = {}
    # Coordinate descent handles the rare case where replacements of different types overlap.
    for _ in range(2):
        for entity_type in sorted(PHASE1_ALLOWED_TYPES):
            candidate_margins = {
                2.0,
                *(
                    margin
                    for candidate_type, margin in _replacement_margin_candidates(
                        examples,
                        probabilities,
                    )
                    if candidate_type == entity_type
                ),
            }
            best_margin = margins[entity_type]
            best_metrics = _metrics_against_gold(
                _select_conservative_examples(examples, probabilities, margins),
                gold,
            )
            best_rank = _selection_metric_rank(best_metrics, best_margin)
            for margin in sorted(candidate_margins, reverse=True):
                trial = {**margins, entity_type: margin}
                metrics = _metrics_against_gold(
                    _select_conservative_examples(
                        examples,
                        probabilities,
                        trial,
                    ),
                    gold,
                )
                rank = _selection_metric_rank(metrics, margin)
                if rank > best_rank:
                    best_margin = margin
                    best_metrics = metrics
                    best_rank = rank
            margins[entity_type] = best_margin
            report[entity_type] = {
                "margin": best_margin,
                "candidate_margin_count": len(candidate_margins),
                "global_metrics_after_step": best_metrics,
            }
    final = _select_conservative_examples(examples, probabilities, margins)
    report["final"] = _metrics_against_gold(final, gold)
    report["base"] = _metrics_against_gold(
        [example for example in examples if example.base_selected],
        gold,
    )
    return margins, report


def _replacement_margin_candidates(
    examples: Sequence[Phase1BoundaryExample],
    probabilities: Sequence[float],
) -> tuple[tuple[str, float], ...]:
    grouped = _group_scored_examples(examples, probabilities)
    output: list[tuple[str, float]] = []
    for values in grouped.values():
        base = _base_family_example(values)
        if base is None:
            continue
        winner = min(
            (
                (example.variant, probability)
                for example, probability in values
            ),
            key=_family_rank_key,
        )
        if winner[0].variant_id == base[0].variant.variant_id:
            continue
        output.append(
            (
                base[0].variant.entity_type,
                max(0.0, winner[1] - base[1]),
            )
        )
    return tuple(output)


def _select_conservative_examples(
    examples: Sequence[Phase1BoundaryExample],
    probabilities: Sequence[float],
    replacement_margins: Mapping[str, float],
) -> tuple[Phase1BoundaryExample, ...]:
    """Correct selected boundaries without adding or deleting proposal-level entities."""

    grouped = _group_scored_examples(examples, probabilities)
    choices: dict[str, tuple[Phase1BoundaryExample, float]] = {}
    replacements: list[
        tuple[float, str, Phase1BoundaryExample, float]
    ] = []
    for family_id, values in grouped.items():
        base_values = [item for item in values if item[0].base_selected]
        for base_example, base_probability in base_values:
            choices[f"{family_id}:{base_example.variant.variant_id}"] = (
                base_example,
                base_probability,
            )
        base = _base_family_example(values)
        if base is None:
            continue
        slot_id = f"{family_id}:{base[0].variant.variant_id}"
        winner_variant, winner_probability = min(
            (
                (example.variant, probability)
                for example, probability in values
            ),
            key=_family_rank_key,
        )
        if winner_variant.variant_id == base[0].variant.variant_id:
            continue
        winner_example = next(
            example
            for example, _ in values
            if example.variant.variant_id == winner_variant.variant_id
        )
        margin = max(0.0, winner_probability - base[1])
        threshold = replacement_margins[base[0].variant.entity_type]
        if margin >= threshold:
            replacements.append(
                (margin, slot_id, winner_example, winner_probability)
            )
    for _, slot_id, replacement, probability in sorted(
        replacements,
        key=lambda item: (
            -item[0],
            _scored_example_sort_key((item[2], item[3])),
        ),
    ):
        if any(
            slot_id != other_slot
            and replacement.variant.document_id
            == other.variant.document_id
            and _overlap(replacement.variant.position, other.variant.position)
            for other_slot, (other, _) in choices.items()
        ):
            continue
        choices[slot_id] = (replacement, probability)
    return tuple(
        example
        for example, _ in sorted(
            choices.values(),
            key=_scored_example_sort_key,
        )
    )


def _group_scored_examples(
    examples: Sequence[Phase1BoundaryExample],
    probabilities: Sequence[float],
) -> dict[str, list[tuple[Phase1BoundaryExample, float]]]:
    grouped: dict[str, list[tuple[Phase1BoundaryExample, float]]] = defaultdict(list)
    for example, probability in zip(examples, probabilities, strict=True):
        grouped[example.variant.family_id].append((example, probability))
    return grouped


def _base_family_example(
    values: Sequence[tuple[Phase1BoundaryExample, float]],
) -> tuple[Phase1BoundaryExample, float] | None:
    base = [item for item in values if item[0].base_selected]
    # Multiple selected foundations in one overlap component represent separate output slots.
    # Their boundaries cannot be changed independently with the current family contract.
    if len(base) != 1:
        return None
    return base[0]


def _selection_metric_rank(
    metrics: Mapping[str, Any],
    margin: float,
) -> tuple[float, float, float, float]:
    return (
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        margin,
    )


def _diagnostic_promotion_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    """Keep manual-gold diagnostics from silently becoming a Round 2 promotion decision."""

    selections = report["development_selection"]
    if not isinstance(selections, Mapping):
        raise ValueError("Boundary report has no development selections")
    baseline = selections["base_proposal_verifier"]
    active = selections["active"]
    if not isinstance(baseline, Mapping) or not isinstance(active, Mapping):
        raise ValueError("Boundary development metrics are malformed")
    f1_delta = float(active["f1"]) - float(baseline["f1"])
    false_positive_delta = int(active["false_positive"]) - int(
        baseline["false_positive"]
    )
    diagnostic_pass = f1_delta >= 0.01 and false_positive_delta <= 0
    return {
        "diagnostic_pass": diagnostic_pass,
        "public_probe_required": True,
        "auto_promote": False,
        "f1_delta": f1_delta,
        "false_positive_delta": false_positive_delta,
        "reason": (
            "manual_gold_is_diagnostic_only"
            if diagnostic_pass
            else "diagnostic_gate_failed"
        ),
    }


def _select_examples(
    examples: Sequence[Phase1BoundaryExample],
    probabilities: Sequence[float],
    thresholds: Mapping[str, float],
    *,
    genre_thresholds: Mapping[tuple[str, str], float] | None = None,
) -> tuple[Phase1BoundaryExample, ...]:
    active_genre_thresholds = genre_thresholds or {}
    winners = _family_winners(
        [
            (example.variant, probability)
            for example, probability in zip(examples, probabilities, strict=True)
        ]
    )
    example_by_id = {
        example.variant.variant_id: example for example in examples
    }
    candidates: list[tuple[Phase1BoundaryExample, float]] = []
    for variant, probability in winners:
        example = example_by_id[variant.variant_id]
        threshold = active_genre_thresholds.get(
            (example.genre, variant.entity_type),
            thresholds[variant.entity_type],
        )
        if probability >= threshold:
            candidates.append((example, probability))
    accepted: list[tuple[Phase1BoundaryExample, float]] = []
    for item in sorted(candidates, key=_scored_example_sort_key):
        example = item[0]
        if any(
            example.variant.document_id == other.variant.document_id
            and _overlap(example.variant.position, other.variant.position)
            for other, _ in accepted
        ):
            continue
        accepted.append(item)
    return tuple(example for example, _ in accepted)


def _family_winners(
    variants: Sequence[tuple[Phase1BoundaryVariant, float]],
) -> tuple[tuple[Phase1BoundaryVariant, float], ...]:
    grouped: dict[str, list[tuple[Phase1BoundaryVariant, float]]] = defaultdict(list)
    for item in variants:
        grouped[item[0].family_id].append(item)
    return tuple(
        min(values, key=_family_rank_key)
        for _, values in sorted(grouped.items())
    )


def _family_rank_key(
    item: tuple[Phase1BoundaryVariant, float],
) -> tuple[Any, ...]:
    variant, probability = item
    return (
        -probability,
        -len(variant.sources),
        -int(variant.position in variant.foundation_spans),
        -(variant.position[1] - variant.position[0]),
        variant.position,
        variant.variant_id,
    )


def _resolve_scored_overlaps(
    items: Sequence[ScoredPhase1BoundaryVariant],
) -> tuple[ScoredPhase1BoundaryVariant, ...]:
    accepted: list[ScoredPhase1BoundaryVariant] = []
    for item in sorted(items, key=_runtime_rank_key):
        if any(
            item.variant.document_id == other.variant.document_id
            and _overlap(item.variant.position, other.variant.position)
            for other in accepted
        ):
            continue
        accepted.append(item)
    return tuple(accepted)


def _conservative_runtime_selection(
    variants: Sequence[tuple[Phase1BoundaryVariant, float]],
    base_selected_keys: set[tuple[tuple[int, int], str]],
    replacement_margins: Mapping[str, float],
    *,
    excluded_variant_ids: set[str],
) -> tuple[set[str], set[str]]:
    """Preserve every base output slot and replace only unambiguous family boundaries."""

    grouped: dict[
        str,
        list[tuple[Phase1BoundaryVariant, float]],
    ] = defaultdict(list)
    for item in variants:
        grouped[item[0].family_id].append(item)
    choices: dict[str, tuple[Phase1BoundaryVariant, float]] = {}
    replacements: list[
        tuple[float, str, Phase1BoundaryVariant, float]
    ] = []
    for family_id, values in grouped.items():
        base = [
            item
            for item in values
            if (item[0].position, item[0].entity_type) in base_selected_keys
        ]
        for variant, probability in base:
            choices[f"{family_id}:{variant.variant_id}"] = (
                variant,
                probability,
            )
        if len(base) != 1:
            continue
        base_variant, base_probability = base[0]
        winner, winner_probability = min(values, key=_family_rank_key)
        if (
            winner.variant_id == base_variant.variant_id
            or winner.variant_id in excluded_variant_ids
        ):
            continue
        margin = max(0.0, winner_probability - base_probability)
        if margin >= replacement_margins[base_variant.entity_type]:
            replacements.append(
                (
                    margin,
                    f"{family_id}:{base_variant.variant_id}",
                    winner,
                    winner_probability,
                )
            )
    accepted_replacements: set[str] = set()
    for _, slot_id, replacement, probability in sorted(
        replacements,
        key=lambda item: (
            -item[0],
            _family_rank_key((item[2], item[3])),
        ),
    ):
        if any(
            slot_id != other_slot
            and replacement.document_id == other.document_id
            and _overlap(replacement.position, other.position)
            for other_slot, (other, _) in choices.items()
        ):
            continue
        choices[slot_id] = (replacement, probability)
        accepted_replacements.add(replacement.variant_id)
    return (
        {variant.variant_id for variant, _ in choices.values()},
        accepted_replacements,
    )


def _selection_metrics(
    selected: Sequence[Phase1BoundaryExample],
    gold_counts: Mapping[tuple[str, str, str | None], int],
    *,
    split: str,
) -> dict[str, Any]:
    scoped = [example for example in selected if example.split == split]
    gold = sum(
        count
        for (count_split, _, genre), count in gold_counts.items()
        if count_split == split and genre is None
    )
    return _metrics_against_gold(scoped, gold)


def _selection_by_genre(
    selected: Sequence[Phase1BoundaryExample],
    gold_counts: Mapping[tuple[str, str, str | None], int],
    *,
    split: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for genre in Phase1GenreBucket:
        scoped = [
            example
            for example in selected
            if example.split == split and example.genre == genre.value
        ]
        gold = sum(
            count
            for (count_split, _, count_genre), count in gold_counts.items()
            if count_split == split and count_genre == genre.value
        )
        if scoped or gold:
            report[genre.value] = _metrics_against_gold(scoped, gold)
    return report


def _metrics_against_gold(
    selected: Sequence[Phase1BoundaryExample],
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
            sorted(Counter(example.error_label for example in selected).items())
        ),
    }


def _family_top1_metrics(
    examples: Sequence[Phase1BoundaryExample],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[Phase1BoundaryExample, float]]] = defaultdict(list)
    for example, probability in zip(examples, probabilities, strict=True):
        grouped[example.variant.family_id].append((example, probability))
    covered = 0
    correct = 0
    for values in grouped.values():
        if not any(example.label for example, _ in values):
            continue
        covered += 1
        winner = min(
            (
                (example.variant, probability)
                for example, probability in values
            ),
            key=_family_rank_key,
        )[0]
        correct += next(
            example.label
            for example, _ in values
            if example.variant.variant_id == winner.variant_id
        )
    return {
        "covered_family_count": covered,
        "correct_family_count": correct,
        "accuracy": correct / covered if covered else 0.0,
    }


def _coverage_metrics(
    examples: Sequence[Phase1BoundaryExample],
    gold_counts: Mapping[tuple[str, str, str | None], int],
    split: str,
) -> dict[str, Any]:
    covered = sum(example.label for example in examples if example.split == split)
    gold = sum(
        count
        for (count_split, _, genre), count in gold_counts.items()
        if count_split == split and genre is None
    )
    return {
        "covered_gold": covered,
        "gold": gold,
        "recall": covered / gold if gold else 0.0,
    }


def _gold_counts(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, str, str | None], int]:
    raw = manifest.get("gold_entity_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("Boundary dataset has no gold entity counts")
    counts: dict[tuple[str, str, str | None], int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise ValueError("Boundary gold counts are malformed")
        parts = key.split(":")
        if len(parts) == 2:
            split, entity_type = parts
            genre: str | None = None
        elif len(parts) == 3:
            split, genre, entity_type = parts
            Phase1GenreBucket(genre)
        else:
            raise ValueError("Boundary gold count key is malformed")
        if split not in {"train", "development"} or entity_type not in PHASE1_ALLOWED_TYPES:
            raise ValueError("Boundary gold count key is invalid")
        counts[(split, entity_type, genre)] = value
    return counts


def _rows_by_document(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        document_id = str(row.get("document_id", ""))
        if not document_id:
            raise ValueError("Boundary proposal row has no document_id")
        grouped[document_id].append(row)
    return {
        document_id: tuple(
            sorted(
                values,
                key=lambda row: (
                    _position(row),
                    str(row.get("type", "")),
                ),
            )
        )
        for document_id, values in grouped.items()
    }


def _serialize_examples(examples: Sequence[Phase1BoundaryExample]) -> str:
    return "".join(
        json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for example in examples
    )


def _example_sort_key(example: Phase1BoundaryExample) -> tuple[Any, ...]:
    variant = example.variant
    return (
        phase1_document_sort_key(variant.document_id),
        variant.position,
        variant.entity_type,
        variant.variant_id,
    )


def _scored_example_sort_key(
    item: tuple[Phase1BoundaryExample, float],
) -> tuple[Any, ...]:
    example, probability = item
    variant = example.variant
    return (
        -probability,
        -PHASE1_TYPE_PRIORITY[variant.entity_type],
        -(variant.position[1] - variant.position[0]),
        phase1_document_sort_key(variant.document_id),
        variant.position,
        variant.variant_id,
    )


def _runtime_rank_key(
    item: ScoredPhase1BoundaryVariant,
) -> tuple[Any, ...]:
    variant = item.variant
    return (
        -item.probability,
        -PHASE1_TYPE_PRIORITY[variant.entity_type],
        -(variant.position[1] - variant.position[0]),
        phase1_document_sort_key(variant.document_id),
        variant.position,
        variant.variant_id,
    )


def _output_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    position = _position(row)
    entity_type = str(row.get("type", ""))
    return (
        position[0],
        position[1],
        -PHASE1_TYPE_PRIORITY.get(entity_type, 0),
        entity_type,
    )


def _position(row: Mapping[str, Any]) -> tuple[int, int]:
    raw = row.get("position")
    if (
        not isinstance(raw, list | tuple)
        or len(raw) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw
        )
        or raw[0] < 0
        or raw[1] <= raw[0]
    ):
        raise ValueError("Boundary row has an invalid position")
    return int(raw[0]), int(raw[1])


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _top_weights(model: SparseLogisticModel, limit: int = 25) -> dict[str, Any]:
    pairs = list(zip(model.feature_names, model.weights, strict=True))
    return {
        "positive": [
            {"feature": name, "weight": weight}
            for name, weight in sorted(
                pairs,
                key=lambda pair: (-pair[1], pair[0]),
            )[:limit]
        ],
        "negative": [
            {"feature": name, "weight": weight}
            for name, weight in sorted(
                pairs,
                key=lambda pair: (pair[1], pair[0]),
            )[:limit]
        ],
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
    values: list[tuple[str, str, float]] = []
    for genre, raw_thresholds in payload.items():
        if not isinstance(genre, str) or not isinstance(raw_thresholds, Mapping):
            raise ValueError("Boundary genre thresholds are invalid")
        Phase1GenreBucket(genre)
        for entity_type, raw_threshold in raw_thresholds.items():
            values.append(
                (
                    genre,
                    str(entity_type),
                    _numeric_threshold(raw_threshold),
                )
            )
    return tuple(sorted(values))


def _numeric_threshold(value: Any) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("Boundary threshold must be finite and numeric")
    return float(value)


def _numeric_margin(value: Any) -> float:
    margin = _numeric_threshold(value)
    if not 0.0 <= margin <= 2.0:
        raise ValueError("Boundary replacement margin must be within [0, 2]")
    return margin


def _binary_value(value: Any, field: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise ValueError(f"Boundary example {field} must be 0 or 1")
    return value


def _boolean_value(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Boundary example {field} must be boolean")
    return value
