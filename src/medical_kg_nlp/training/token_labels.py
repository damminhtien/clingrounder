"""Fast-tokenizer BIO projection and framework-neutral exact-span metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from medical_kg_nlp.training.span_dataset import SpanTrainingEntity, SpanTrainingRecord

__all__ = [
    "FastTokenizerPort",
    "TokenBoundaryAlignmentError",
    "TokenizedTrainingWindow",
    "compute_bio_span_metrics",
    "decode_bio_spans",
    "project_record_to_token_windows",
]

_IGNORE_INDEX = -100


class FastTokenizerPort(Protocol):
    """Minimal tokenizer surface required by the model-neutral projection layer."""

    is_fast: bool

    def __call__(self, text: str, **kwargs: object) -> Mapping[str, Any]: ...


class TokenBoundaryAlignmentError(ValueError):
    """Raised when a character annotation cannot be represented by tokenizer spans."""


@dataclass(frozen=True)
class TokenizedTrainingWindow:
    """One tokenizer window with exactly one owner for each supervised entity."""

    record_id: str
    window_index: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    token_type_ids: tuple[int, ...] | None = None

    def to_model_dict(self) -> dict[str, list[int]]:
        """Return only tensor-ready fields consumed by token-classification models."""

        output = {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "labels": list(self.labels),
        }
        if self.token_type_ids is not None:
            output["token_type_ids"] = list(self.token_type_ids)
        return output


def project_record_to_token_windows(
    record: SpanTrainingRecord,
    tokenizer: FastTokenizerPort,
    label_vocabulary: Sequence[str],
    *,
    max_length: int,
    stride: int,
) -> tuple[TokenizedTrainingWindow, ...]:
    """Tokenize one record and project character labels without duplicating entities.

    Overflow windows often contain the same entity. Each entity is assigned to the
    window with the most context; overlapping copies in other windows are ignored so
    long records do not over-weight labels near a window boundary.
    """

    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("Token-classifier training requires a fast tokenizer")
    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    if stride < 0 or stride >= max_length - 2:
        raise ValueError("stride must fit inside max_length")
    label_to_id = _label_to_id(label_vocabulary)

    encoded = tokenizer(
        record.text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        padding=False,
    )
    input_rows = _integer_rows(encoded.get("input_ids"), "input_ids")
    offset_rows = _offset_rows(encoded.get("offset_mapping"))
    if len(input_rows) != len(offset_rows):
        raise ValueError("Tokenizer input and offset window counts differ")
    attention_rows = _optional_integer_rows(encoded.get("attention_mask"), len(input_rows))
    token_type_rows = _optional_integer_rows(encoded.get("token_type_ids"), len(input_rows))
    if attention_rows is None:
        attention_rows = tuple(tuple(1 for _ in row) for row in input_rows)

    owners = _assign_entity_owners(record.entities, offset_rows)
    windows: list[TokenizedTrainingWindow] = []
    for window_index, (input_ids, attention_mask, offsets) in enumerate(
        zip(input_rows, attention_rows, offset_rows, strict=True)
    ):
        if len(input_ids) != len(attention_mask) or len(input_ids) != len(offsets):
            raise ValueError("Tokenizer fields have inconsistent token counts")
        labels = [
            _IGNORE_INDEX if start == end else label_to_id["O"]
            for start, end in offsets
        ]
        for entity_index, entity in enumerate(record.entities):
            token_indices = _overlapping_token_indices(entity, offsets)
            if not token_indices:
                continue
            if owners[entity_index] != window_index:
                for token_index in token_indices:
                    labels[token_index] = _IGNORE_INDEX
                continue
            _validate_entity_alignment(entity, offsets, token_indices)
            for entity_token_index, token_index in enumerate(token_indices):
                prefix = "B" if entity_token_index == 0 else "I"
                labels[token_index] = label_to_id[f"{prefix}-{entity.label}"]

        token_type_ids = None
        if token_type_rows is not None:
            token_type_ids = token_type_rows[window_index]
            if len(token_type_ids) != len(input_ids):
                raise ValueError("token_type_ids has an inconsistent token count")
        windows.append(
            TokenizedTrainingWindow(
                record_id=record.record_id,
                window_index=window_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=tuple(labels),
                offsets=offsets,
                token_type_ids=token_type_ids,
            )
        )

    if not windows:
        raise ValueError(f"Tokenizer produced no windows for record {record.record_id!r}")
    return tuple(windows)


def decode_bio_spans(
    label_ids: Sequence[int],
    label_vocabulary: Sequence[str],
) -> tuple[tuple[int, int, str], ...]:
    """Decode BIO IDs into token-index spans, treating malformed I-tags as new spans."""

    spans: list[tuple[int, int, str]] = []
    active_start: int | None = None
    active_label: str | None = None

    def flush(end: int) -> None:
        nonlocal active_start, active_label
        if active_start is not None and active_label is not None:
            spans.append((active_start, end, active_label))
        active_start = None
        active_label = None

    for index, label_id in enumerate(label_ids):
        if label_id == _IGNORE_INDEX:
            flush(index)
            continue
        if not 0 <= label_id < len(label_vocabulary):
            raise ValueError(f"Unknown token label ID {label_id}")
        raw_label = label_vocabulary[label_id]
        if raw_label == "O":
            flush(index)
            continue
        prefix, separator, entity_label = raw_label.partition("-")
        if not separator or prefix not in {"B", "I"} or not entity_label:
            raise ValueError(f"Invalid BIO label {raw_label!r}")
        continues = prefix == "I" and active_start is not None and active_label == entity_label
        if not continues:
            flush(index)
            active_start = index
            active_label = entity_label
    flush(len(label_ids))
    return tuple(spans)


def compute_bio_span_metrics(
    predicted_label_ids: Sequence[Sequence[int]],
    gold_label_ids: Sequence[Sequence[int]],
    label_vocabulary: Sequence[str],
) -> dict[str, float]:
    """Compute exact micro span/type metrics across tokenized evaluation windows."""

    if len(predicted_label_ids) != len(gold_label_ids):
        raise ValueError("Prediction and gold window counts differ")
    predicted: set[tuple[int, int, int, str]] = set()
    gold: set[tuple[int, int, int, str]] = set()
    for row_index, (predicted_row, gold_row) in enumerate(
        zip(predicted_label_ids, gold_label_ids, strict=True)
    ):
        if len(predicted_row) != len(gold_row):
            raise ValueError("Prediction and gold token counts differ")
        masked_predictions = [
            _IGNORE_INDEX if gold_id == _IGNORE_INDEX else predicted_id
            for predicted_id, gold_id in zip(predicted_row, gold_row, strict=True)
        ]
        predicted.update(
            (row_index, start, end, label)
            for start, end, label in decode_bio_spans(
                masked_predictions,
                label_vocabulary,
            )
        )
        gold.update(
            (row_index, start, end, label)
            for start, end, label in decode_bio_spans(gold_row, label_vocabulary)
        )

    true_positive = len(predicted & gold)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    precision = true_positive / (true_positive + false_positive) if predicted else 0.0
    recall = true_positive / (true_positive + false_negative) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
        "span_true_positive": float(true_positive),
        "span_false_positive": float(false_positive),
        "span_false_negative": float(false_negative),
    }


def _assign_entity_owners(
    entities: Sequence[SpanTrainingEntity],
    offset_rows: Sequence[Sequence[tuple[int, int]]],
) -> dict[int, int]:
    owners: dict[int, int] = {}
    for entity_index, entity in enumerate(entities):
        candidates: list[tuple[int, int]] = []
        alignment_errors: list[TokenBoundaryAlignmentError] = []
        for window_index, offsets in enumerate(offset_rows):
            token_indices = _overlapping_token_indices(entity, offsets)
            if not token_indices:
                continue
            try:
                _validate_entity_alignment(entity, offsets, token_indices)
            except TokenBoundaryAlignmentError as error:
                alignment_errors.append(error)
                continue
            visible = [(start, end) for start, end in offsets if start != end]
            if not visible:
                continue
            left_context = entity.start - visible[0][0]
            right_context = visible[-1][1] - entity.end
            candidates.append((min(left_context, right_context), window_index))
        if not candidates:
            if alignment_errors:
                raise alignment_errors[0]
            raise TokenBoundaryAlignmentError(
                f"Annotation {entity.annotation_id!r} is not contained in any token window"
            )
        # MODEL: maximal context reduces edge artifacts; lower index makes ties stable.
        _, owner = max(candidates, key=lambda item: (item[0], -item[1]))
        owners[entity_index] = owner
    return owners


def _overlapping_token_indices(
    entity: SpanTrainingEntity,
    offsets: Sequence[tuple[int, int]],
) -> list[int]:
    return [
        index
        for index, (start, end) in enumerate(offsets)
        if start != end and start < entity.end and entity.start < end
    ]


def _validate_entity_alignment(
    entity: SpanTrainingEntity,
    offsets: Sequence[tuple[int, int]],
    token_indices: Sequence[int],
) -> None:
    first_start = offsets[token_indices[0]][0]
    last_end = offsets[token_indices[-1]][1]
    fully_contained = all(
        entity.start <= offsets[index][0] and offsets[index][1] <= entity.end
        for index in token_indices
    )
    if first_start != entity.start or last_end != entity.end or not fully_contained:
        raise TokenBoundaryAlignmentError(
            "Tokenizer cannot preserve exact boundary for annotation "
            f"{entity.annotation_id!r} at [{entity.start}, {entity.end})"
        )


def _label_to_id(label_vocabulary: Sequence[str]) -> dict[str, int]:
    if not label_vocabulary or label_vocabulary[0] != "O":
        raise ValueError("BIO vocabulary must start with O")
    if len(set(label_vocabulary)) != len(label_vocabulary):
        raise ValueError("BIO vocabulary contains duplicate labels")
    return {label: index for index, label in enumerate(label_vocabulary)}


def _integer_rows(value: object, field_name: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"Tokenizer {field_name} must be a non-empty sequence")
    raw_rows: Sequence[object]
    first = value[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        raw_rows = value
    else:
        raw_rows = (value,)
    rows: list[tuple[int, ...]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes)):
            raise ValueError(f"Tokenizer {field_name} row must be a sequence")
        row = tuple(_strict_int(item, field_name) for item in raw_row)
        rows.append(row)
    return tuple(rows)


def _optional_integer_rows(
    value: object,
    expected_rows: int,
) -> tuple[tuple[int, ...], ...] | None:
    if value is None:
        return None
    rows = _integer_rows(value, "optional model input")
    if len(rows) != expected_rows:
        raise ValueError("Tokenizer optional input window count differs")
    return rows


def _offset_rows(value: object) -> tuple[tuple[tuple[int, int], ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("Tokenizer offset_mapping must be a non-empty sequence")
    first = value[0]
    if _is_offset_pair(first):
        raw_rows: Sequence[object] = (value,)
    else:
        raw_rows = value
    rows: list[tuple[tuple[int, int], ...]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes)):
            raise ValueError("Tokenizer offset row must be a sequence")
        row: list[tuple[int, int]] = []
        for raw_pair in raw_row:
            if not _is_offset_pair(raw_pair):
                raise ValueError("Tokenizer offsets must be integer pairs")
            assert isinstance(raw_pair, Sequence)
            start = _strict_int(raw_pair[0], "offset")
            end = _strict_int(raw_pair[1], "offset")
            if start < 0 or end < start:
                raise ValueError("Tokenizer offset is invalid")
            row.append((start, end))
        rows.append(tuple(row))
    return tuple(rows)


def _is_offset_pair(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _strict_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Tokenizer {field_name} values must be integers")
    return value
