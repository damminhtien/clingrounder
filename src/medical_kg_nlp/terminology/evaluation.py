"""Task-neutral retrieval evaluation for pinned terminology repositories."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Any

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository

__all__ = [
    "TerminologyQuery",
    "evaluate_terminology_queries",
    "load_terminology_queries",
]

_ABSTENTION_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


@dataclass(frozen=True)
class TerminologyQuery:
    """One mention with dictionary-constrained expected codes."""

    query_id: str
    mention: str
    entity_type: EntityType
    code_system: CodeSystem
    expected_codes: tuple[str, ...]
    slices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.query_id or not self.mention.strip() or not self.expected_codes:
            raise ValueError("Terminology queries require ID, mention, and expected codes")
        if self.code_system == CodeSystem.NONE:
            raise ValueError("Terminology queries cannot target the NONE code system")
        if any(not value.strip() for value in self.slices) or len(self.slices) != len(
            set(self.slices)
        ):
            raise ValueError("Terminology query slices must be unique non-empty strings")


def load_terminology_queries(path: str | Path) -> tuple[TerminologyQuery, ...]:
    """Load neutral JSONL queries and reject duplicate IDs."""

    queries: list[TerminologyQuery] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{source}:{line_number}: expected JSON object")
            codes = raw.get("expected_codes")
            if not isinstance(codes, list) or not codes:
                raise ValueError(
                    f"{source}:{line_number}: expected_codes must be a non-empty array"
                )
            queries.append(
                TerminologyQuery(
                    query_id=str(raw["query_id"]),
                    mention=str(raw["mention"]),
                    entity_type=EntityType(str(raw["entity_type"])),
                    code_system=CodeSystem(str(raw["code_system"])),
                    expected_codes=tuple(str(code) for code in codes),
                    slices=_optional_string_tuple(raw, "slices", source, line_number),
                )
            )
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Terminology query IDs must be unique")
    return tuple(queries)


def evaluate_terminology_queries(
    repository: TerminologyRepository,
    queries: Sequence[TerminologyQuery],
    *,
    modes: Sequence[str] = ("exact", "toneless", "search"),
    limit: int = 20,
) -> dict[str, Any]:
    """Evaluate rank and latency while applying query type/system filters before limit."""

    if not queries:
        raise ValueError("At least one terminology query is required")
    if limit < 1:
        raise ValueError("limit must be positive")
    unsupported = sorted(set(modes) - {"exact", "toneless", "search"})
    if unsupported:
        raise ValueError(f"Unsupported terminology query modes: {unsupported}")

    mode_reports: dict[str, Any] = {}
    for mode in modes:
        rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        for query in queries:
            started = perf_counter()
            concepts, scores = _query(repository, query, mode=mode, limit=limit)
            latencies.append((perf_counter() - started) * 1000.0)
            codes, code_scores = _ordered_code_scores(concepts, scores)
            expected = set(query.expected_codes)
            rank = next(
                (index for index, code in enumerate(codes, start=1) if code in expected),
                None,
            )
            expected_score = max(
                (score for code, score in zip(codes, code_scores) if code in expected),
                default=None,
            )
            rows.append(
                {
                    "query_id": query.query_id,
                    "mention": query.mention,
                    "entity_type": query.entity_type.value,
                    "code_system": query.code_system.value,
                    "expected_codes": list(query.expected_codes),
                    "candidate_codes": codes,
                    "candidate_scores": code_scores,
                    "candidate_count": len(codes),
                    "expected_rank": rank,
                    "expected_score": expected_score,
                    "top_score": code_scores[0] if code_scores else None,
                    "top1_correct": rank == 1,
                    "slices": list(query.slices),
                }
            )
        mode_reports[mode] = {
            "metrics": _rank_metrics(rows),
            "abstention_curve": _abstention_curve(rows),
            "slice_metrics": _slice_metrics(rows),
            "latency_ms": _latency_metrics(latencies),
            "errors": [row for row in rows if row["expected_rank"] is None],
        }

    expected_code_pairs = {
        (query.code_system, code) for query in queries for code in query.expected_codes
    }
    unknown_pairs = sorted(
        (system.value, code)
        for system, code in expected_code_pairs
        if repository.get_by_code(system, code) is None
    )
    return {
        "schema_version": "terminology-retrieval-evaluation.v3",
        "query_count": len(queries),
        "unique_mention_count": len({query.mention.casefold() for query in queries}),
        "entity_type_counts": dict(
            sorted(Counter(query.entity_type.value for query in queries).items())
        ),
        "code_system_counts": dict(
            sorted(Counter(query.code_system.value for query in queries).items())
        ),
        "slice_counts": dict(
            sorted(Counter(value for query in queries for value in query.slices).items())
        ),
        "unknown_expected_code_count": len(unknown_pairs),
        "unknown_expected_codes": [
            {"code_system": system, "code": code} for system, code in unknown_pairs
        ],
        "limit": limit,
        "modes": mode_reports,
    }


def _query(
    repository: TerminologyRepository,
    query: TerminologyQuery,
    *,
    mode: str,
    limit: int,
) -> tuple[list[ConceptEntry], list[float]]:
    if mode == "exact":
        concepts = repository.exact_lookup(
            query.mention,
            entity_type=query.entity_type,
            code_systems=(query.code_system,),
            limit=limit,
        )
        return concepts, [1.0] * len(concepts)
    if mode == "toneless":
        concepts = repository.toneless_lookup(
            query.mention,
            entity_type=query.entity_type,
            code_systems=(query.code_system,),
            limit=limit,
        )
        return concepts, [0.92] * len(concepts)
    hits = repository.search_scored(
        query.mention,
        entity_type=query.entity_type,
        code_systems=(query.code_system,),
        limit=limit,
    )
    return [hit.entry for hit in hits], [hit.score for hit in hits]


def _ordered_code_scores(
    concepts: Sequence[ConceptEntry],
    scores: Sequence[float],
) -> tuple[list[str], list[float]]:
    if len(concepts) != len(scores):
        raise ValueError("Terminology concepts and scores must have equal length")
    output: list[str] = []
    output_scores: list[float] = []
    seen: set[str] = set()
    for concept, score in zip(concepts, scores):
        code = concept.code
        if code is None or code in seen:
            continue
        seen.add(code)
        output.append(str(code))
        output_scores.append(score)
    return output, output_scores


def _rank_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    ranks = [int(row["expected_rank"]) for row in rows if row["expected_rank"] is not None]
    total = len(rows)
    return {
        "hit_at_1": sum(rank <= 1 for rank in ranks) / total,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / total,
        "recall_at_10": sum(rank <= 10 for rank in ranks) / total,
        "recall_at_20": sum(rank <= 20 for rank in ranks) / total,
        "mrr": sum(1.0 / rank for rank in ranks) / total,
        "matched_query_count": len(ranks),
        "missing_query_count": total - len(ranks),
        "empty_candidate_count": sum(not row["candidate_codes"] for row in rows),
        "ambiguous_candidate_count": sum(len(row["candidate_codes"]) > 1 for row in rows),
    }


def _abstention_curve(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int]]:
    total = len(rows)
    output: list[dict[str, float | int]] = []
    for threshold in _ABSTENTION_THRESHOLDS:
        emitted = [
            row
            for row in rows
            if row.get("top_score") is not None
            and float(row["top_score"]) >= threshold
        ]
        correct = sum(bool(row.get("top1_correct")) for row in emitted)
        output.append(
            {
                "threshold": threshold,
                "emitted_query_count": len(emitted),
                "correct_query_count": correct,
                "coverage": len(emitted) / total,
                "precision": correct / len(emitted) if emitted else 0.0,
                "recall": correct / total,
            }
        )
    return output


def _slice_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    slices = sorted(
        {
            str(value)
            for row in rows
            for value in row.get("slices", [])
            if str(value)
        }
    )
    return {
        slice_name: _rank_metrics(
            [row for row in rows if slice_name in row.get("slices", [])]
        )
        for slice_name in slices
    }


def _latency_metrics(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": fmean(ordered),
        "p50": median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


def _optional_string_tuple(
    raw: Mapping[str, object],
    field: str,
    path: Path,
    line_number: int,
) -> tuple[str, ...]:
    value = raw.get(field, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{path}:{line_number}: {field} must be a string array")
    return tuple(str(item).strip() for item in value)
