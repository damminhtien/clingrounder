"""Project model token labels onto validated raw-text entity spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from clingrounder.schema.types import EntityType

__all__ = [
    "ProjectedEntity",
    "ProjectedSourceEntity",
    "TokenPrediction",
    "project_bio_predictions",
    "project_source_bio_predictions",
]

_BEGIN = frozenset({"B"})
_INSIDE = frozenset({"I"})
_END = frozenset({"E", "L"})
_SINGLE = frozenset({"S", "U"})
_OUTSIDE = frozenset({"O", "PAD", "IGNORE"})


@dataclass(frozen=True)
class TokenPrediction:
    """One token label and confidence expressed in original-text coordinates."""

    start: int
    end: int
    label: str
    score: float


@dataclass(frozen=True)
class ProjectedEntity:
    """A model entity before pipeline-specific ID assignment."""

    span: tuple[int, int]
    entity_type: EntityType
    confidence: float


@dataclass(frozen=True)
class ProjectedSourceEntity:
    """A raw model-taxonomy span retained before any task-specific crosswalk."""

    span: tuple[int, int]
    source_label: str
    confidence: float


def project_bio_predictions(
    source_text: str,
    predictions: Sequence[TokenPrediction],
    *,
    label_map: Mapping[str, EntityType] | None = None,
    confidence_thresholds: Mapping[EntityType, float] | None = None,
    default_confidence_threshold: float = 0.0,
) -> list[ProjectedEntity]:
    """Merge BIO/BIOES tokens and reject projections not backed by raw text.

    Overflow windows can emit the same token more than once. Exact token spans are
    deduplicated by confidence before sequence decoding, then entity-level overlaps
    are resolved by confidence and span length.
    """

    if not 0.0 <= default_confidence_threshold <= 1.0:
        raise ValueError("default_confidence_threshold must be between 0 and 1")
    mapped_labels = label_map or {}
    thresholds = confidence_thresholds or {}
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds.values()):
        raise ValueError("confidence thresholds must be between 0 and 1")
    best_by_span: dict[tuple[int, int], TokenPrediction] = {}
    for prediction in predictions:
        if not 0 <= prediction.start < prediction.end <= len(source_text):
            continue
        current = best_by_span.get((prediction.start, prediction.end))
        if current is None or _prediction_order(prediction) > _prediction_order(current):
            best_by_span[(prediction.start, prediction.end)] = prediction

    decoded: list[ProjectedEntity] = []
    active: list[TokenPrediction] = []
    active_type: EntityType | None = None

    def flush() -> None:
        nonlocal active, active_type
        if active and active_type is not None:
            start = active[0].start
            end = active[-1].end
            # INVARIANT: model entities are always slices of the untouched source text.
            if source_text[start:end]:
                decoded.append(
                    ProjectedEntity(
                        span=(start, end),
                        entity_type=active_type,
                        confidence=sum(item.score for item in active) / len(active),
                    )
                )
        active = []
        active_type = None

    for prediction in sorted(
        best_by_span.values(),
        key=lambda item: (item.start, item.end, -item.score, item.label),
    ):
        prefix, entity_type = _decode_label(prediction.label, mapped_labels)
        if entity_type is None or prefix in _OUTSIDE:
            flush()
            continue
        if prefix in _SINGLE:
            flush()
            active = [prediction]
            active_type = entity_type
            flush()
            continue
        if prefix in _BEGIN:
            flush()
            active = [prediction]
            active_type = entity_type
            continue

        continues_active = (
            active
            and active_type == entity_type
            and prediction.start >= active[-1].end
        )
        if not continues_active:
            flush()
            active = [prediction]
            active_type = entity_type
        else:
            active.append(prediction)
        if prefix in _END:
            flush()

    flush()
    # MODEL: threshold before overlap resolution so a rejected high-scoring type cannot suppress
    # a valid proposal of another type that occupies the same source region.
    accepted = [
        entity
        for entity in decoded
        if entity.confidence
        >= thresholds.get(entity.entity_type, default_confidence_threshold)
    ]
    return _resolve_entity_overlaps(accepted)


def project_source_bio_predictions(
    source_text: str,
    predictions: Sequence[TokenPrediction],
    *,
    confidence_thresholds: Mapping[str, float] | None = None,
    default_confidence_threshold: float = 0.0,
) -> list[ProjectedSourceEntity]:
    """Decode BIO/BIOES labels while preserving a model's source taxonomy.

    This projection is intentionally task-neutral. A downstream benchmark adapter may
    map one source label to multiple compatible target types, but the model prediction
    itself remains auditable and cannot silently become target-task gold.
    """

    if not 0.0 <= default_confidence_threshold <= 1.0:
        raise ValueError("default confidence threshold must be between 0 and 1")
    thresholds = dict(confidence_thresholds or {})
    if any(
        not label.strip() or not 0.0 <= threshold <= 1.0
        for label, threshold in thresholds.items()
    ):
        raise ValueError("source-label confidence thresholds are invalid")

    best_by_span: dict[tuple[int, int], TokenPrediction] = {}
    for prediction in predictions:
        if not 0 <= prediction.start < prediction.end <= len(source_text):
            continue
        current = best_by_span.get((prediction.start, prediction.end))
        if current is None or _prediction_order(prediction) > _prediction_order(current):
            best_by_span[(prediction.start, prediction.end)] = prediction

    decoded: list[ProjectedSourceEntity] = []
    active: list[TokenPrediction] = []
    active_label: str | None = None

    def flush() -> None:
        nonlocal active, active_label
        if active and active_label is not None:
            start = active[0].start
            end = active[-1].end
            confidence = sum(item.score for item in active) / len(active)
            if (
                source_text[start:end]
                and confidence
                >= thresholds.get(active_label, default_confidence_threshold)
            ):
                decoded.append(
                    ProjectedSourceEntity(
                        span=(start, end),
                        source_label=active_label,
                        confidence=confidence,
                    )
                )
        active = []
        active_label = None

    for prediction in sorted(
        best_by_span.values(),
        key=lambda item: (item.start, item.end, -item.score, item.label),
    ):
        prefix, source_label = _decode_unmapped_label(prediction.label)
        if source_label is None or prefix in _OUTSIDE:
            flush()
            continue
        if prefix in _SINGLE:
            flush()
            active = [prediction]
            active_label = source_label
            flush()
            continue
        if prefix in _BEGIN:
            flush()
            active = [prediction]
            active_label = source_label
            continue
        continues_active = (
            active
            and active_label == source_label
            and prediction.start >= active[-1].end
        )
        if not continues_active:
            flush()
            active = [prediction]
            active_label = source_label
        else:
            active.append(prediction)
        if prefix in _END:
            flush()
    flush()
    return _resolve_source_entity_overlaps(decoded)


def _decode_label(
    raw_label: str,
    label_map: Mapping[str, EntityType],
) -> tuple[str, EntityType | None]:
    label = raw_label.strip()
    if not label:
        return "O", None
    prefix = "S"
    entity_label = label
    for separator in ("-", "_"):
        head, found, tail = label.partition(separator)
        if found and head.upper() in _BEGIN | _INSIDE | _END | _SINGLE | _OUTSIDE:
            prefix = head.upper()
            entity_label = tail
            break
    if label.upper() in _OUTSIDE:
        return label.upper(), None
    mapped = label_map.get(entity_label) or label_map.get(label)
    if mapped is not None:
        return prefix, mapped
    try:
        return prefix, EntityType(entity_label.upper())
    except ValueError:
        return prefix, None


def _decode_unmapped_label(raw_label: str) -> tuple[str, str | None]:
    label = raw_label.strip()
    if not label or label.upper() in _OUTSIDE:
        return label.upper() or "O", None
    prefix = "S"
    source_label = label
    for separator in ("-", "_"):
        head, found, tail = label.partition(separator)
        if found and head.upper() in _BEGIN | _INSIDE | _END | _SINGLE:
            prefix = head.upper()
            source_label = tail
            break
    if not source_label:
        raise ValueError(f"Source BIO label has no entity type: {raw_label!r}")
    return prefix, source_label


def _prediction_order(prediction: TokenPrediction) -> tuple[float, str]:
    return prediction.score, prediction.label


def _resolve_entity_overlaps(entities: list[ProjectedEntity]) -> list[ProjectedEntity]:
    ranked = sorted(
        entities,
        key=lambda item: (
            -item.confidence,
            -(item.span[1] - item.span[0]),
            item.span[0],
            item.entity_type.value,
        ),
    )
    selected: list[ProjectedEntity] = []
    for candidate in ranked:
        if any(
            candidate.span[0] < existing.span[1]
            and existing.span[0] < candidate.span[1]
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: (item.span[0], item.span[1], item.entity_type.value))


def _resolve_source_entity_overlaps(
    entities: list[ProjectedSourceEntity],
) -> list[ProjectedSourceEntity]:
    ranked = sorted(
        entities,
        key=lambda item: (
            -item.confidence,
            -(item.span[1] - item.span[0]),
            item.span[0],
            item.source_label,
        ),
    )
    selected: list[ProjectedSourceEntity] = []
    for candidate in ranked:
        if any(
            candidate.span[0] < existing.span[1]
            and existing.span[0] < candidate.span[1]
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return sorted(
        selected,
        key=lambda item: (item.span[0], item.span[1], item.source_label),
    )
