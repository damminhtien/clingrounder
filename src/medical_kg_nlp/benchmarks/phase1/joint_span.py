"""Joint span/type lattice contracts and deterministic global resolution for Phase 1.

The module deliberately separates candidate generation from learned scoring. Every candidate is a
raw source substring produced by bounded generators, and the verifier only chooses among those
alternatives. This makes the learned stage auditable and preserves offset safety independently of
the underlying transformer.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from medical_kg_nlp.benchmarks.phase1.boundary_variants import (
    BoundaryErrorLabel,
    Phase1BoundaryVariant,
    boundary_cross_encoder_text,
    generate_phase1_boundary_variants,
    label_phase1_boundary_variant,
)
from medical_kg_nlp.benchmarks.phase1.proposal_conflict_graph import (
    Phase1ConflictNode,
    build_phase1_conflict_graph,
    select_maximum_utility_nodes,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import (
    Phase1GenreBucket,
    ProposalSourceRole,
    extract_phase1_proposal_context,
    phase1_genre_bucket,
)
from medical_kg_nlp.ner.dictionary_matcher import DictionaryMatcher
from medical_kg_nlp.ner.document_structure import (
    DocumentStructure,
    DocumentStructureAnalyzer,
)
from medical_kg_nlp.ontology.phase1 import PHASE1_ALLOWED_TYPES

__all__ = [
    "Phase1JointSpanCandidate",
    "Phase1JointSpanLabel",
    "Phase1JointSpanPrediction",
    "Phase1JointSpanSelectionPolicy",
    "Phase1JointSpanVerifierPort",
    "ScoredPhase1JointSpanCandidate",
    "generate_phase1_joint_span_lattice",
    "label_phase1_joint_span_candidate",
    "resolve_phase1_joint_span_lattice",
]

_EXACT_LABEL_BY_TYPE = {
    "CHẨN_ĐOÁN": "EXACT_DISEASE",
    "TRIỆU_CHỨNG": "EXACT_SYMPTOM",
    "THUỐC": "EXACT_DRUG",
    "TÊN_XÉT_NGHIỆM": "EXACT_LAB_TEST",
    "KẾT_QUẢ_XÉT_NGHIỆM": "EXACT_LAB_RESULT",
}


class Phase1JointSpanLabel(StrEnum):
    """Mutually exclusive target classes for a span/type verification example."""

    EXACT_DISEASE = "EXACT_DISEASE"
    EXACT_SYMPTOM = "EXACT_SYMPTOM"
    EXACT_DRUG = "EXACT_DRUG"
    EXACT_LAB_TEST = "EXACT_LAB_TEST"
    EXACT_LAB_RESULT = "EXACT_LAB_RESULT"
    TOO_SHORT = "TOO_SHORT"
    TOO_LONG = "TOO_LONG"
    SPURIOUS = "SPURIOUS"


@dataclass(frozen=True, slots=True)
class Phase1JointSpanCandidate:
    """One model-ready candidate with its raw-offset identity and rendered joint input."""

    variant: Phase1BoundaryVariant
    genre: str
    section: str
    cross_encoder_text: str

    def __post_init__(self) -> None:
        Phase1GenreBucket(self.genre)
        if not self.section.strip():
            raise ValueError("Joint span candidate section must be non-empty")
        if not self.cross_encoder_text.strip():
            raise ValueError("Joint span candidate input must be non-empty")

    @property
    def expected_exact_label(self) -> Phase1JointSpanLabel:
        """Return the only exact class compatible with this candidate's entity type."""

        return Phase1JointSpanLabel(_EXACT_LABEL_BY_TYPE[self.variant.entity_type])

    def to_dict(self) -> dict[str, Any]:
        """Serialize enough provenance to retrain or audit an exact decision."""

        return {
            **self.variant.to_dict(),
            "genre": self.genre,
            "section": self.section,
            "cross_encoder_text": self.cross_encoder_text,
        }


@dataclass(frozen=True, slots=True)
class Phase1JointSpanPrediction:
    """One normalized verifier distribution for a lattice candidate."""

    variant_id: str
    probabilities: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.variant_id.strip():
            raise ValueError("Joint span prediction must identify a candidate")
        labels = tuple(label for label, _ in self.probabilities)
        if set(labels) != {label.value for label in Phase1JointSpanLabel}:
            raise ValueError("Joint span prediction must define every label exactly once")
        if len(set(labels)) != len(labels):
            raise ValueError("Joint span prediction labels must be unique")
        values = tuple(value for _, value in self.probabilities)
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Joint span probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("Joint span probabilities must sum to one")

    def probability(self, label: Phase1JointSpanLabel) -> float:
        """Return one class probability without exposing mutable implementation state."""

        return dict(self.probabilities)[label.value]

    def exact_probability(self, candidate: Phase1JointSpanCandidate) -> float:
        """Return probability that both this candidate's span and type are exact."""

        return self.probability(candidate.expected_exact_label)


@runtime_checkable
class Phase1JointSpanVerifierPort(Protocol):
    """Score bounded candidate spans with a pinned multi-class verifier."""

    @property
    def provenance(self) -> str:
        """Return a pinned model/artifact identifier for traces."""

    def predict(
        self,
        candidates: Sequence[Phase1JointSpanCandidate],
    ) -> Sequence[Phase1JointSpanPrediction]:
        """Return one normalized distribution for every input candidate."""


@dataclass(frozen=True, slots=True)
class Phase1JointSpanSelectionPolicy:
    """Explicit genre/type thresholds and false-positive utility cost.

    ``global_per_type`` is an intentional fallback, not a clinical threshold reused for a
    question-answer document. The resolver records that fallback for each scored candidate.
    """

    type_thresholds: tuple[tuple[str, float], ...]
    genre_type_thresholds: tuple[tuple[str, str, float], ...] = ()
    false_positive_cost: float = 1.0

    def __post_init__(self) -> None:
        expected = set(PHASE1_ALLOWED_TYPES)
        observed = {entity_type for entity_type, _ in self.type_thresholds}
        if observed != expected or len(observed) != len(self.type_thresholds):
            raise ValueError("Joint span policy must define one threshold for every entity type")
        if any(not 0.0 < value < 1.0 for _, value in self.type_thresholds):
            raise ValueError("Joint span type thresholds must be within (0, 1)")
        genre_keys: set[tuple[str, str]] = set()
        for genre, entity_type, value in self.genre_type_thresholds:
            Phase1GenreBucket(genre)
            if entity_type not in expected or not 0.0 < value < 1.0:
                raise ValueError("Joint span genre threshold is invalid")
            if (genre, entity_type) in genre_keys:
                raise ValueError("Joint span genre thresholds must be unique")
            genre_keys.add((genre, entity_type))
        if not math.isfinite(self.false_positive_cost) or self.false_positive_cost <= 0.0:
            raise ValueError("Joint span false_positive_cost must be finite and positive")

    @classmethod
    def conservative_default(cls) -> "Phase1JointSpanSelectionPolicy":
        """Return a traceable baseline until OOF calibration creates pinned thresholds."""

        return cls(tuple((entity_type, 0.5) for entity_type in sorted(PHASE1_ALLOWED_TYPES)))

    def threshold_for(self, candidate: Phase1JointSpanCandidate) -> tuple[float, str]:
        """Resolve an operating point without silently borrowing clinical calibration."""

        by_genre = {
            (genre, entity_type): threshold
            for genre, entity_type, threshold in self.genre_type_thresholds
        }
        key = (candidate.genre, candidate.variant.entity_type)
        if key in by_genre:
            return by_genre[key], "genre_type"
        return dict(self.type_thresholds)[candidate.variant.entity_type], "global_per_type"


@dataclass(frozen=True, slots=True)
class ScoredPhase1JointSpanCandidate:
    """Auditable score, expected gain, and final global-resolution decision."""

    candidate: Phase1JointSpanCandidate
    prediction: Phase1JointSpanPrediction
    exact_probability: float
    threshold: float
    threshold_source: str
    expected_exact_gain: float
    selected: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "probabilities": dict(self.prediction.probabilities),
            "exact_probability": self.exact_probability,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "expected_exact_gain": self.expected_exact_gain,
            "selected": self.selected,
            "rejection_reason": self.rejection_reason,
        }


def generate_phase1_joint_span_lattice(
    document_id: str,
    source_text: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_roles: Mapping[str, ProposalSourceRole | str],
    dictionary_matcher: DictionaryMatcher | None = None,
    structure: DocumentStructure | None = None,
) -> tuple[Phase1JointSpanCandidate, ...]:
    """Generate all bounded raw-offset alternatives for one document.

    INVARIANT: no candidate survives unless ``source_text[start:end] == candidate.text``.
    The shared lattice includes Qwen, XLM-R, rule, medication-parser, trie, coordination, and
    bounded grammar variants through their source/generator provenance.
    """

    active_structure = structure or DocumentStructureAnalyzer().analyze(source_text)
    variants = generate_phase1_boundary_variants(
        document_id,
        source_text,
        rows,
        source_roles=source_roles,
        dictionary_matcher=dictionary_matcher,
        structure=active_structure,
    )
    candidates: list[Phase1JointSpanCandidate] = []
    for variant in variants:
        start, end = variant.position
        if source_text[start:end] != variant.text:
            raise ValueError("Joint span lattice candidate violates the raw-offset invariant")
        context = extract_phase1_proposal_context(
            variant.to_proposal_row(),
            source_text,
            structure=active_structure,
        )
        candidates.append(
            Phase1JointSpanCandidate(
                variant=variant,
                genre=phase1_genre_bucket(active_structure.genre).value,
                section=context.section,
                cross_encoder_text=boundary_cross_encoder_text(
                    variant,
                    source_text,
                    structure=active_structure,
                ),
            )
        )
    return tuple(sorted(candidates, key=_candidate_sort_key))


def label_phase1_joint_span_candidate(
    candidate: Phase1JointSpanCandidate,
    gold_entities: Sequence[Mapping[str, Any]],
) -> Phase1JointSpanLabel:
    """Derive the multi-class exact/boundary target from one reviewed gold document."""

    boundary_label = label_phase1_boundary_variant(candidate.variant, gold_entities)
    if boundary_label is BoundaryErrorLabel.CORRECT:
        return candidate.expected_exact_label
    if boundary_label is BoundaryErrorLabel.TOO_SHORT:
        return Phase1JointSpanLabel.TOO_SHORT
    if boundary_label is BoundaryErrorLabel.TOO_LONG:
        return Phase1JointSpanLabel.TOO_LONG
    return Phase1JointSpanLabel.SPURIOUS


def resolve_phase1_joint_span_lattice(
    candidates: Sequence[Phase1JointSpanCandidate],
    verifier: Phase1JointSpanVerifierPort,
    *,
    policy: Phase1JointSpanSelectionPolicy,
) -> tuple[dict[str, list[dict[str, Any]]], tuple[ScoredPhase1JointSpanCandidate, ...]]:
    """Select a maximum expected-gain, non-overlapping entity set across all proposal sources."""

    predictions = tuple(verifier.predict(candidates))
    if len(predictions) != len(candidates):
        raise ValueError("Joint span verifier must return one prediction per candidate")
    by_variant_id = {prediction.variant_id: prediction for prediction in predictions}
    if len(by_variant_id) != len(predictions):
        raise ValueError("Joint span verifier returned duplicate candidate predictions")
    expected_ids = {candidate.variant.variant_id for candidate in candidates}
    if set(by_variant_id) != expected_ids:
        raise ValueError("Joint span verifier predictions do not match the candidate lattice")

    admissible: list[tuple[Phase1JointSpanCandidate, float, float, str]] = []
    rejected: dict[str, str] = {}
    for candidate in candidates:
        prediction = by_variant_id[candidate.variant.variant_id]
        exact_probability = prediction.exact_probability(candidate)
        threshold, threshold_source = policy.threshold_for(candidate)
        expected_gain = exact_probability - policy.false_positive_cost * (1.0 - exact_probability)
        if exact_probability < threshold:
            rejected[candidate.variant.variant_id] = "below_calibrated_threshold"
            continue
        if expected_gain <= 0.0:
            rejected[candidate.variant.variant_id] = "non_positive_expected_gain"
            continue
        admissible.append((candidate, exact_probability, expected_gain, threshold_source))

    graph = build_phase1_conflict_graph(
        tuple(
            Phase1ConflictNode(
                node_id=candidate.variant.variant_id,
                document_id=candidate.variant.document_id,
                span=candidate.variant.position,
                entity_type=candidate.variant.entity_type,
                probability=probability,
                source_count=len(candidate.variant.sources),
                decision_threshold=policy.threshold_for(candidate)[0],
                utility_override=expected_gain,
            )
            for candidate, probability, expected_gain, _ in admissible
        )
    )
    selected_ids = {node.node_id for node in select_maximum_utility_nodes(graph)}
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scored: list[ScoredPhase1JointSpanCandidate] = []
    for candidate in candidates:
        prediction = by_variant_id[candidate.variant.variant_id]
        exact_probability = prediction.exact_probability(candidate)
        threshold, threshold_source = policy.threshold_for(candidate)
        expected_gain = exact_probability - policy.false_positive_cost * (1.0 - exact_probability)
        selected = candidate.variant.variant_id in selected_ids
        reason = rejected.get(candidate.variant.variant_id)
        if not selected and reason is None:
            reason = "conflicts_with_higher_expected_gain"
        scored.append(
            ScoredPhase1JointSpanCandidate(
                candidate=candidate,
                prediction=prediction,
                exact_probability=exact_probability,
                threshold=threshold,
                threshold_source=threshold_source,
                expected_exact_gain=expected_gain,
                selected=selected,
                rejection_reason=reason,
            )
        )
        if selected:
            variant = candidate.variant
            output[variant.document_id].append(
                {
                    "document_id": variant.document_id,
                    "text": variant.text,
                    "type": variant.entity_type,
                    "position": [variant.position[0], variant.position[1]],
                    "sources": list(variant.sources),
                    "source_count": len(variant.sources),
                    "proposal_id": variant.variant_id,
                    "joint_expected_exact_gain": expected_gain,
                    "joint_model_provenance": verifier.provenance,
                }
            )
    for document_id, rows in output.items():
        rows.sort(key=lambda row: (_output_position(row), str(row["type"])))
    return dict(output), tuple(sorted(scored, key=_scored_candidate_sort_key))


def _candidate_sort_key(candidate: Phase1JointSpanCandidate) -> tuple[Any, ...]:
    return (
        candidate.variant.document_id,
        candidate.variant.position,
        candidate.variant.entity_type,
        candidate.variant.variant_id,
    )


def _scored_candidate_sort_key(
    scored: ScoredPhase1JointSpanCandidate,
) -> tuple[Any, ...]:
    return _candidate_sort_key(scored.candidate)


def _output_position(row: Mapping[str, Any]) -> tuple[int, int]:
    """Read one emitted raw position without importing a legacy resolver helper."""

    raw = row.get("position")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("Joint span output position must be a two-item list")
    start, end = raw
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("Joint span output position must contain integers")
    return start, end
