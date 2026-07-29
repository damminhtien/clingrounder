"""Comparable scoring for Phase 1 rule, model, LLM, and support proposals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.model_selection import (
    load_internal_predictions,
)
from medical_kg_nlp.benchmarks.phase1.phase1 import (
    prediction_to_phase1_entities,
)
from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import (
    load_phase1_output_source,
)
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import (
    Phase1ReviewedCorpus,
)
from medical_kg_nlp.benchmarks.phase1.split_contract import (
    phase1_document_sort_key,
)
from medical_kg_nlp.ontology.phase1 import (
    PHASE1_ALLOWED_TYPES,
    PHASE1_TYPE_BY_ENTITY_TYPE,
)
from medical_kg_nlp.utils.hashing import sha256_file

__all__ = [
    "Phase1ProposalSource",
    "Phase1SourceSemantics",
    "build_phase1_proposal_source_report",
    "load_compatible_phase1_source",
    "load_internal_phase1_source",
    "load_target_phase1_source",
    "source_path_fingerprint",
    "write_phase1_proposal_source_report",
]

_REPORT_SCHEMA = "phase1-proposal-source-report.v1"


class Phase1SourceSemantics(StrEnum):
    """Whether rows assert a target label or only a set of compatible labels."""

    TARGET = "target"
    COMPATIBLE = "compatible"


@dataclass(frozen=True, slots=True)
class Phase1ProposalSource:
    """One frozen proposal source plus honest label semantics."""

    name: str
    rows_by_document: Mapping[str, Sequence[Mapping[str, Any]]]
    semantics: Phase1SourceSemantics = Phase1SourceSemantics.TARGET
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Proposal source name must be non-empty")


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int
    entity_type: str
    text: str

    @property
    def position(self) -> tuple[int, int]:
        return self.start, self.end


@dataclass(frozen=True, slots=True)
class _CompatibleSpan:
    start: int
    end: int
    text: str
    allowed_types: frozenset[str]
    source_label: str

    @property
    def position(self) -> tuple[int, int]:
        return self.start, self.end


def load_target_phase1_source(
    path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load an ordinary flat Phase 1 directory or ZIP."""

    return load_phase1_output_source(path)


def load_internal_phase1_source(
    path: str | Path,
    source_text_by_document: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Project internal prediction JSONL into flat target labels."""

    predictions = load_internal_predictions(path)
    rows: dict[str, list[dict[str, Any]]] = {}
    for document_id, source_text in source_text_by_document.items():
        prediction = predictions.get(document_id)
        if prediction is None:
            continue
        projected = prediction_to_phase1_entities(
            prediction,
            source_text=source_text,
            assertion_policy="empty",
            candidate_policy="empty",
        )
        eligible_entities = [
            entity
            for entity in prediction.entities
            if entity.type in PHASE1_TYPE_BY_ENTITY_TYPE
        ]
        if len(projected) != len(eligible_entities):
            raise ValueError(
                f"Internal projection count mismatch for document {document_id}"
            )
        for row, entity in zip(projected, eligible_entities, strict=True):
            # MODEL: confidence is proposal evidence, not an exported Phase 1 field.
            row["confidence"] = entity.confidence
        rows[document_id] = projected
    return rows


def load_compatible_phase1_source(
    path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load support rows whose duplicated target types encode compatibility."""

    source = Path(path)
    if source.is_dir() or source.suffix.lower() == ".zip":
        return load_phase1_output_source(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: dict[str, list[dict[str, Any]]] = {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{source}:{line_number}: expected an object")
            document_id = str(payload.get("document_id", ""))
            entities = payload.get("entities")
            if not document_id or not isinstance(entities, list):
                raise ValueError(
                    f"{source}:{line_number}: expected document_id and entities"
                )
            rows[document_id] = [
                dict(entity)
                for entity in entities
                if isinstance(entity, Mapping)
            ]
    return rows


def build_phase1_proposal_source_report(
    sources: Sequence[Phase1ProposalSource],
    corpus: Phase1ReviewedCorpus,
    *,
    corpus_fingerprint_sha256: str,
) -> dict[str, Any]:
    """Score every source on identical available train/development documents.

    INVARIANT: a source may cover one complete split or both; partial split coverage is rejected.
    Compatible source labels are grouped before scoring, so VietMed ``DISEASESYMTOM`` is counted
    once and never masquerades as two target predictions.
    """

    if not sources:
        raise ValueError("At least one proposal source is required")
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("Proposal source names must be unique")
    source_reports: dict[str, Any] = {}
    all_errors: list[dict[str, Any]] = []
    for source in sources:
        split_reports: dict[str, Any] = {}
        for split in ("train", "development"):
            document_ids = corpus.document_ids(split)
            available = {
                document_id
                for document_id in document_ids
                if document_id in source.rows_by_document
            }
            if not available:
                continue
            if available != set(document_ids):
                missing = sorted(
                    set(document_ids) - available,
                    key=phase1_document_sort_key,
                )
                raise ValueError(
                    f"Source {source.name!r} partially covers {split}: missing={missing}"
                )
            gold = {
                document_id: corpus.gold_rows[document_id]
                for document_id in document_ids
            }
            predictions = {
                document_id: source.rows_by_document[document_id]
                for document_id in document_ids
            }
            if source.semantics is Phase1SourceSemantics.COMPATIBLE:
                metrics, errors = _compatible_source_metrics(
                    gold,
                    predictions,
                    corpus.source_texts,
                    source_name=source.name,
                    split=split,
                )
            else:
                metrics, errors = _target_source_metrics(
                    gold,
                    predictions,
                    corpus.source_texts,
                    source_name=source.name,
                    split=split,
                )
            split_reports[split] = metrics
            all_errors.extend(errors)
        if not split_reports:
            raise ValueError(
                f"Source {source.name!r} covers neither train nor development"
            )
        source_reports[source.name] = {
            "semantics": source.semantics.value,
            "provenance": dict(source.provenance or {}),
            "splits": split_reports,
        }
    return {
        "schema_version": _REPORT_SCHEMA,
        "holdout_opened": False,
        "corpus_fingerprint_sha256": corpus_fingerprint_sha256,
        "sources": source_reports,
        "error_count": len(all_errors),
        "error_counts": dict(
            sorted(Counter(str(row["error_type"]) for row in all_errors).items())
        ),
        "errors": all_errors,
    }


def write_phase1_proposal_source_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    """Write machine-readable metrics, detailed errors, and a compact comparison."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    errors = report.get("errors")
    error_rows = errors if isinstance(errors, list) else []
    metrics = dict(report)
    metrics.pop("errors", None)
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "errors.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in error_rows
        ),
        encoding="utf-8",
    )
    (output / "summary.md").write_text(_summary_markdown(metrics), encoding="utf-8")


def source_path_fingerprint(path: str | Path) -> str:
    """Hash one file or a deterministic tree of files for report provenance."""

    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    for child in sorted(item for item in source.rglob("*") if item.is_file()):
        digest.update(child.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _target_source_metrics(
    gold_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    pred_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    source_text_by_document: Mapping[str, str],
    *,
    source_name: str,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_gold: list[_Span] = []
    all_pred: list[_Span] = []
    errors: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    per_type_gold: dict[str, list[_Span]] = {
        entity_type: [] for entity_type in PHASE1_ALLOWED_TYPES
    }
    per_type_pred: dict[str, list[_Span]] = {
        entity_type: [] for entity_type in PHASE1_ALLOWED_TYPES
    }
    exact_total = boundary_total = 0
    for document_id in sorted(gold_by_document, key=phase1_document_sort_key):
        source_text = source_text_by_document[document_id]
        gold = _target_spans(gold_by_document[document_id], source_text)
        pred = _target_spans(pred_by_document[document_id], source_text)
        all_gold.extend(gold)
        all_pred.extend(pred)
        for span in gold:
            per_type_gold[span.entity_type].append(span)
        for span in pred:
            per_type_pred[span.entity_type].append(span)
        document_errors, counts, exact_count, boundary_count = _classify_target_errors(
            document_id,
            gold,
            pred,
            source_name=source_name,
            split=split,
        )
        errors.extend(document_errors)
        error_counts.update(counts)
        exact_total += exact_count
        boundary_total += boundary_count

    exact_metrics = _prf(
        exact_total,
        len(all_pred) - exact_total,
        len(all_gold) - exact_total,
    )
    relaxed_tp = exact_total + boundary_total
    relaxed = _prf(
        relaxed_tp,
        len(all_pred) - relaxed_tp,
        len(all_gold) - relaxed_tp,
    )
    return (
        {
            "document_count": len(gold_by_document),
            "gold_count": len(all_gold),
            "proposal_count": len(all_pred),
            "exact": exact_metrics,
            "relaxed_same_type_overlap": relaxed,
            "per_type": {
                entity_type: _type_metrics(
                    per_type_gold[entity_type],
                    per_type_pred[entity_type],
                )
                for entity_type in sorted(PHASE1_ALLOWED_TYPES)
            },
            "error_counts": dict(sorted(error_counts.items())),
        },
        errors,
    )


def _compatible_source_metrics(
    gold_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    pred_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    source_text_by_document: Mapping[str, str],
    *,
    source_name: str,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exact_tp = overlap_tp = gold_count = proposal_count = 0
    unambiguous_gold: dict[str, tuple[Mapping[str, Any], ...]] = {}
    unambiguous_pred: dict[str, tuple[dict[str, Any], ...]] = {}
    source_label_counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    for document_id in sorted(gold_by_document, key=phase1_document_sort_key):
        source_text = source_text_by_document[document_id]
        gold = _target_spans(gold_by_document[document_id], source_text)
        compatible = _compatible_spans(pred_by_document[document_id], source_text)
        gold_count += len(gold)
        proposal_count += len(compatible)
        exact_pairs = _maximum_pairs(
            len(gold),
            len(compatible),
            lambda left, right: (
                gold[left].position == compatible[right].position
                and gold[left].entity_type in compatible[right].allowed_types
            ),
        )
        overlap_pairs = _maximum_pairs(
            len(gold),
            len(compatible),
            lambda left, right: (
                gold[left].entity_type in compatible[right].allowed_types
                and _overlap(gold[left].position, compatible[right].position)
            ),
        )
        exact_tp += len(exact_pairs)
        overlap_tp += len(overlap_pairs)
        source_label_counts.update(row.source_label for row in compatible)
        for index, compatible_row in enumerate(compatible):
            if index not in {right for _, right in overlap_pairs}:
                errors.append(
                    _error_row(
                        source_name,
                        split,
                        document_id,
                        "compatible_spurious",
                        prediction=_compatible_to_dict(compatible_row),
                    )
                )
        for index, gold_row in enumerate(gold):
            if index not in {left for left, _ in overlap_pairs}:
                errors.append(
                    _error_row(
                        source_name,
                        split,
                        document_id,
                        "compatible_missing",
                        gold=_span_to_dict(gold_row),
                    )
                )
        unambiguous = [row for row in compatible if len(row.allowed_types) == 1]
        unambiguous_gold[document_id] = tuple(gold_by_document[document_id])
        unambiguous_pred[document_id] = tuple(
            {
                "text": row.text,
                "type": next(iter(row.allowed_types)),
                "position": [row.start, row.end],
                "assertions": [],
                "candidates": [],
            }
            for row in unambiguous
        )
    strict_metrics, strict_errors = _target_source_metrics(
        unambiguous_gold,
        unambiguous_pred,
        source_text_by_document,
        source_name=f"{source_name}:unambiguous",
        split=split,
    )
    errors.extend(strict_errors)
    return (
        {
            "document_count": len(gold_by_document),
            "gold_count": gold_count,
            "proposal_count": proposal_count,
            "compatible_exact": _prf(
                exact_tp,
                proposal_count - exact_tp,
                gold_count - exact_tp,
            ),
            "compatible_overlap": _prf(
                overlap_tp,
                proposal_count - overlap_tp,
                gold_count - overlap_tp,
            ),
            "source_label_counts": dict(sorted(source_label_counts.items())),
            "target_label_note": (
                "Compatible labels are evidence sets, not direct Phase 1 target predictions."
            ),
            "unambiguous_projection": strict_metrics,
        },
        errors,
    )


def _classify_target_errors(
    document_id: str,
    gold: Sequence[_Span],
    pred: Sequence[_Span],
    *,
    source_name: str,
    split: str,
) -> tuple[list[dict[str, Any]], Counter[str], int, int]:
    unmatched_gold = set(range(len(gold)))
    unmatched_pred = set(range(len(pred)))
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    exact_pairs = _stage_pairs(
        gold,
        pred,
        unmatched_gold,
        unmatched_pred,
        lambda left, right: left.position == right.position
        and left.entity_type == right.entity_type,
    )
    type_pairs = _stage_pairs(
        gold,
        pred,
        unmatched_gold,
        unmatched_pred,
        lambda left, right: left.position == right.position
        and left.entity_type != right.entity_type,
    )
    boundary_pairs = _stage_pairs(
        gold,
        pred,
        unmatched_gold,
        unmatched_pred,
        lambda left, right: left.entity_type == right.entity_type
        and _overlap(left.position, right.position),
    )
    boundary_type_pairs = _stage_pairs(
        gold,
        pred,
        unmatched_gold,
        unmatched_pred,
        lambda left, right: left.entity_type != right.entity_type
        and _overlap(left.position, right.position),
    )
    for error_type, pairs in (
        ("type_confusion", type_pairs),
        ("boundary", boundary_pairs),
        ("boundary_type_confusion", boundary_type_pairs),
    ):
        counts[error_type] += len(pairs)
        errors.extend(
            _error_row(
                source_name,
                split,
                document_id,
                error_type,
                gold=_span_to_dict(gold[left]),
                prediction=_span_to_dict(pred[right]),
            )
            for left, right in pairs
        )
    counts["missing"] += len(unmatched_gold)
    counts["spurious"] += len(unmatched_pred)
    errors.extend(
        _error_row(
            source_name,
            split,
            document_id,
            "missing",
            gold=_span_to_dict(gold[index]),
        )
        for index in sorted(unmatched_gold)
    )
    errors.extend(
        _error_row(
            source_name,
            split,
            document_id,
            "spurious",
            prediction=_span_to_dict(pred[index]),
        )
        for index in sorted(unmatched_pred)
    )
    return errors, counts, len(exact_pairs), len(boundary_pairs)


def _stage_pairs(
    gold: Sequence[_Span],
    pred: Sequence[_Span],
    unmatched_gold: set[int],
    unmatched_pred: set[int],
    predicate: Any,
) -> list[tuple[int, int]]:
    left_ids = sorted(unmatched_gold)
    right_ids = sorted(unmatched_pred)
    local_pairs = _maximum_pairs(
        len(left_ids),
        len(right_ids),
        lambda left, right: predicate(gold[left_ids[left]], pred[right_ids[right]]),
    )
    pairs = [(left_ids[left], right_ids[right]) for left, right in local_pairs]
    unmatched_gold.difference_update(left for left, _ in pairs)
    unmatched_pred.difference_update(right for _, right in pairs)
    return pairs


def _type_metrics(gold: Sequence[_Span], pred: Sequence[_Span]) -> dict[str, Any]:
    exact_tp = len(
        _maximum_pairs(
            len(gold),
            len(pred),
            lambda left, right: gold[left].position == pred[right].position,
        )
    )
    overlap_tp = len(
        _maximum_pairs(
            len(gold),
            len(pred),
            lambda left, right: _overlap(gold[left].position, pred[right].position),
        )
    )
    return {
        "gold_count": len(gold),
        "proposal_count": len(pred),
        "exact": _prf(exact_tp, len(pred) - exact_tp, len(gold) - exact_tp),
        "overlap": _prf(
            overlap_tp,
            len(pred) - overlap_tp,
            len(gold) - overlap_tp,
        ),
    }


def _target_spans(
    rows: Sequence[Mapping[str, Any]],
    source_text: str,
) -> list[_Span]:
    spans: dict[tuple[int, int, str], _Span] = {}
    for row in rows:
        start, end, text = _validated_row(row, source_text)
        entity_type = str(row.get("type", ""))
        if entity_type not in PHASE1_ALLOWED_TYPES:
            raise ValueError(f"Unsupported Phase 1 type {entity_type!r}")
        spans[(start, end, entity_type)] = _Span(
            start=start,
            end=end,
            entity_type=entity_type,
            text=text,
        )
    return sorted(
        spans.values(),
        key=lambda row: (row.start, row.end, row.entity_type),
    )


def _compatible_spans(
    rows: Sequence[Mapping[str, Any]],
    source_text: str,
) -> list[_CompatibleSpan]:
    groups: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        start, end, text = _validated_row(row, source_text)
        entity_type = str(row.get("type", ""))
        if entity_type not in PHASE1_ALLOWED_TYPES:
            raise ValueError(f"Unsupported compatible target {entity_type!r}")
        source_label = str(row.get("source_label", "unknown"))
        key = start, end, source_label
        group = groups.setdefault(
            key,
            {"text": text, "allowed_types": set()},
        )
        group["allowed_types"].add(entity_type)
    return [
        _CompatibleSpan(
            start=start,
            end=end,
            text=str(group["text"]),
            allowed_types=frozenset(group["allowed_types"]),
            source_label=source_label,
        )
        for (start, end, source_label), group in sorted(groups.items())
    ]


def _validated_row(
    row: Mapping[str, Any],
    source_text: str,
) -> tuple[int, int, str]:
    position = row.get("position")
    text = row.get("text")
    if (
        not isinstance(position, list)
        or len(position) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in position
        )
        or not isinstance(text, str)
        or not text
    ):
        raise ValueError("Proposal has invalid text/position fields")
    start, end = position
    if start < 0 or end <= start or end > len(source_text):
        raise ValueError("Proposal position is outside source text")
    if source_text[start:end] != text:
        raise ValueError("Proposal text does not match raw source offset")
    return start, end, text


def _maximum_pairs(
    left_count: int,
    right_count: int,
    predicate: Any,
) -> list[tuple[int, int]]:
    """Return deterministic maximum-cardinality bipartite matches."""

    adjacency = [
        [right for right in range(right_count) if predicate(left, right)]
        for left in range(left_count)
    ]
    right_to_left: dict[int, int] = {}

    def augment(left: int, seen: set[int]) -> bool:
        for right in adjacency[left]:
            if right in seen:
                continue
            seen.add(right)
            previous = right_to_left.get(right)
            if previous is None or augment(previous, seen):
                right_to_left[right] = left
                return True
        return False

    for left in range(left_count):
        augment(left, set())
    return sorted((left, right) for right, left in right_to_left.items())


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _span_to_dict(row: _Span) -> dict[str, Any]:
    return {
        "text": row.text,
        "type": row.entity_type,
        "position": [row.start, row.end],
    }


def _compatible_to_dict(row: _CompatibleSpan) -> dict[str, Any]:
    return {
        "text": row.text,
        "allowed_types": sorted(row.allowed_types),
        "source_label": row.source_label,
        "position": [row.start, row.end],
    }


def _error_row(
    source: str,
    split: str,
    document_id: str,
    error_type: str,
    *,
    gold: Mapping[str, Any] | None = None,
    prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "split": split,
        "document_id": document_id,
        "error_type": error_type,
        "gold": dict(gold) if gold is not None else None,
        "prediction": dict(prediction) if prediction is not None else None,
    }


def _summary_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 1 Proposal Source Report",
        "",
        "Holdout opened: **no**",
        "",
        "| Source | Semantics | Split | Exact/compatible F1 | Recall | Proposals |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    sources = report.get("sources")
    if isinstance(sources, Mapping):
        for name, raw_source in sorted(sources.items()):
            if not isinstance(raw_source, Mapping):
                continue
            semantics = str(raw_source.get("semantics", ""))
            splits = raw_source.get("splits")
            if not isinstance(splits, Mapping):
                continue
            for split, raw_metrics in sorted(splits.items()):
                if not isinstance(raw_metrics, Mapping):
                    continue
                key = "compatible_exact" if semantics == "compatible" else "exact"
                prf = raw_metrics.get(key)
                if not isinstance(prf, Mapping):
                    continue
                lines.append(
                    f"| {name} | {semantics} | {split} | "
                    f"{float(prf.get('f1', 0.0)):.4f} | "
                    f"{float(prf.get('recall', 0.0)):.4f} | "
                    f"{int(raw_metrics.get('proposal_count', 0))} |"
                )
    return "\n".join(lines) + "\n"
