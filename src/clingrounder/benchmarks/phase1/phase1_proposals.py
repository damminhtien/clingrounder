"""Align independent Phase 1 entity proposals without deciding which one wins.

The matrix is the audit boundary between proposal generation and learned fusion. It preserves
source confidence and source-task labels so the downstream verifier can estimate proposal
correctness instead of relying on source-count bonuses.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from clingrounder.utils.text import normalize_for_match

__all__ = [
    "build_phase1_proposal_matrix",
    "proposal_consensus_keys",
    "write_phase1_proposal_matrix",
]


def build_phase1_proposal_matrix(
    sources: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    source_text_by_doc: Mapping[str, str],
    *,
    source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Align independent Phase 1 proposals without adjudicating them."""
    if len(sources) < 2:
        raise ValueError("Proposal matrix requires at least two independent sources.")
    source_names = tuple(sorted(sources))
    if any(not name.strip() for name in source_names):
        raise ValueError("Proposal source names must be non-empty.")
    metadata = {name: dict((source_metadata or {}).get(name, {})) for name in source_names}

    document_ids = sorted(
        set(source_text_by_doc) | {doc_id for rows in sources.values() for doc_id in rows},
        key=_document_sort_key,
    )
    matrix_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    for document_id in document_ids:
        source_text = source_text_by_doc.get(document_id, "")
        exact_groups: dict[tuple[int, int, str], dict[str, Any]] = {}
        for source_name in source_names:
            seen_in_source: set[tuple[int, int, str]] = set()
            for row_index, row in enumerate(sources[source_name].get(document_id, [])):
                parsed = _parse_proposal(row, source_text)
                if parsed is None:
                    invalid_rows.append(
                        {
                            "document_id": document_id,
                            "source": source_name,
                            "row_index": row_index,
                            "reason": "invalid_schema_or_offset",
                            "proposal": row,
                        }
                    )
                    continue
                start, end, entity_type, text = parsed
                key = (start, end, entity_type)
                if key in seen_in_source:
                    _merge_source_evidence(
                        exact_groups[key]["source_evidence"],
                        source_name,
                        row,
                    )
                    continue
                seen_in_source.add(key)
                source_counts[source_name] += 1
                group = exact_groups.setdefault(
                    key,
                    {
                        "document_id": document_id,
                        "position": [start, end],
                        "text": text,
                        "normalized_mention": normalize_for_match(text),
                        "type": entity_type,
                        "sources": [],
                        "source_evidence": {},
                    },
                )
                group["sources"].append(source_name)
                _merge_source_evidence(
                    group["source_evidence"],
                    source_name,
                    row,
                )

        groups = list(exact_groups.values())
        for group in groups:
            start, end = group["position"]
            group_sources = set(group["sources"])
            overlap: list[dict[str, Any]] = []
            type_conflicts: list[dict[str, Any]] = []
            for other in groups:
                if other is group:
                    continue
                other_start, other_end = other["position"]
                other_sources = set(other["sources"])
                if start == other_start and end == other_end and group["type"] != other["type"]:
                    type_conflicts.append(
                        {
                            "type": other["type"],
                            "sources": sorted(other_sources),
                        }
                    )
                    continue
                if group["type"] != other["type"] or end <= other_start or other_end <= start:
                    continue
                if other_sources <= group_sources:
                    continue
                overlap.append(
                    {
                        "position": other["position"],
                        "text": other["text"],
                        "sources": sorted(other_sources),
                    }
                )
            if type_conflicts:
                status = "type_conflict"
            elif len(group_sources) >= 2:
                status = "exact_agreement"
            elif overlap:
                status = "overlap_agreement"
            else:
                status = "source_only"
            group["sources"] = sorted(group_sources)
            group["source_evidence"] = {
                source: group["source_evidence"][source]
                for source in sorted(group["source_evidence"])
            }
            group["source_count"] = len(group_sources)
            group["all_source_agreement"] = len(group_sources) == len(source_names)
            group["status"] = status
            group["overlap_agreements"] = sorted(
                overlap,
                key=lambda item: (item["position"][0], item["position"][1], item["text"]),
            )
            group["type_conflicts"] = sorted(type_conflicts, key=lambda item: item["type"])
            group["context_pattern"] = _context_pattern(source_text, start, end)
            group["proposal_id"] = _proposal_id(document_id, start, end, str(group["type"]))
            matrix_rows.append(group)

    matrix_rows.sort(key=_matrix_sort_key)
    review_queue = _build_review_queue(matrix_rows, len(source_names))
    status_counts = Counter(str(row["status"]) for row in matrix_rows)
    exact_consensus = sum(1 for row in matrix_rows if int(row["source_count"]) >= 2)
    return {
        "schema_version": "phase1-proposal-matrix.v3",
        "source_names": list(source_names),
        "source_metadata": metadata,
        "summary": {
            "document_count": len(document_ids),
            "proposal_group_count": len(matrix_rows),
            "invalid_proposal_count": len(invalid_rows),
            "exact_consensus_count": exact_consensus,
            "all_source_agreement_count": sum(
                1 for row in matrix_rows if bool(row["all_source_agreement"])
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "source_proposal_counts": dict(sorted(source_counts.items())),
            "review_group_count": len(review_queue),
        },
        "matrix": matrix_rows,
        "invalid_proposals": invalid_rows,
        "review_queue": review_queue,
        "blind_documents": [
            {
                "document_id": document_id,
                "text": source_text_by_doc[document_id],
                "instruction": (
                    "Propose Phase 1 entities from raw text only. Do not consult pipeline or Qwen outputs. "
                    "Return text, type, assertions, candidates, and [start, end) position."
                ),
            }
            for document_id in document_ids
            if document_id in source_text_by_doc
        ],
    }


def proposal_consensus_keys(
    report: Mapping[str, Any],
    *,
    minimum_sources: int = 2,
) -> set[tuple[str, int, int, str]]:
    keys: set[tuple[str, int, int, str]] = set()
    matrix = report.get("matrix")
    if not isinstance(matrix, list):
        return keys
    for row in matrix:
        if not isinstance(row, Mapping) or int(row.get("source_count", 0)) < minimum_sources:
            continue
        position = row.get("position")
        if not _valid_position(position):
            continue
        assert isinstance(position, list)
        keys.add(
            (
                str(row.get("document_id", "")),
                int(position[0]),
                int(position[1]),
                str(row.get("type", "")),
            )
        )
    return keys


def write_phase1_proposal_matrix(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", report.get("summary", {}))
    _write_json(output / "source_metadata.json", report.get("source_metadata", {}))
    _write_jsonl(output / "proposal_matrix.jsonl", report.get("matrix", []))
    _write_jsonl(output / "invalid_proposals.jsonl", report.get("invalid_proposals", []))
    _write_jsonl(output / "codex_blind_queue.jsonl", report.get("blind_documents", []))
    _write_review_csv(output / "review_queue.csv", report.get("review_queue", []))


def _build_review_queue(matrix_rows: list[dict[str, Any]], source_count: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in matrix_rows:
        if bool(row["all_source_agreement"]):
            continue
        signature = "+".join(row["sources"])
        key = (
            str(row["normalized_mention"]),
            str(row["type"]),
            str(row["status"]),
            signature,
        )
        group = groups.setdefault(
            key,
            {
                "normalized_mention": key[0],
                "type": key[1],
                "status": key[2],
                "sources": signature,
                "occurrence_count": 0,
                "document_support": set(),
                "context_patterns": Counter(),
                "examples": [],
                "review_decision": "",
                "review_notes": "",
            },
        )
        group["occurrence_count"] += 1
        group["document_support"].add(str(row["document_id"]))
        group["context_patterns"][str(row["context_pattern"])] += 1
        if len(group["examples"]) < 3:
            group["examples"].append(
                {
                    "document_id": row["document_id"],
                    "text": row["text"],
                    "position": row["position"],
                }
            )
    rows: list[dict[str, Any]] = []
    for group in groups.values():
        rows.append(
            {
                **{key: value for key, value in group.items() if key not in {"document_support", "context_patterns"}},
                "document_support": len(group["document_support"]),
                "context_patterns": [
                    {"pattern": pattern, "count": count}
                    for pattern, count in group["context_patterns"].most_common(3)
                ],
                "missing_source_count": source_count - len(str(group["sources"]).split("+")),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["document_support"]),
            -int(row["occurrence_count"]),
            str(row["status"]),
            str(row["normalized_mention"]),
        ),
    )


def _parse_proposal(
    row: Mapping[str, Any], source_text: str
) -> tuple[int, int, str, str] | None:
    text = row.get("text")
    entity_type = row.get("type")
    position = row.get("position")
    if not isinstance(text, str) or not text or not isinstance(entity_type, str):
        return None
    if not _valid_position(position):
        return None
    assert isinstance(position, list)
    start, end = position
    if start < 0 or end > len(source_text) or source_text[start:end] != text:
        return None
    return start, end, entity_type, text


def _merge_source_evidence(
    evidence_by_source: dict[str, dict[str, Any]],
    source_name: str,
    row: Mapping[str, Any],
) -> None:
    """Merge duplicate exact proposals while retaining the strongest source evidence."""

    confidence = _optional_confidence(row)
    source_label = row.get("source_label")
    if source_label is not None and (
        not isinstance(source_label, str) or not source_label.strip()
    ):
        raise ValueError("Proposal source_label must be a non-empty string")
    current = evidence_by_source.setdefault(
        source_name,
        {
            "confidence": None,
            "source_labels": [],
            "support_only": False,
        },
    )
    previous = current["confidence"]
    if confidence is not None and (previous is None or confidence > previous):
        current["confidence"] = confidence
    if isinstance(source_label, str):
        current["source_labels"] = sorted(
            {*current["source_labels"], source_label}
        )
    current["support_only"] = bool(current["support_only"]) or bool(
        row.get("support_only", False)
    )


def _optional_confidence(row: Mapping[str, Any]) -> float | None:
    raw = row.get("confidence", row.get("score"))
    if raw is None:
        return None
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        raise ValueError("Proposal confidence must be numeric")
    confidence = float(raw)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("Proposal confidence must be finite and within [0, 1]")
    return confidence


def _context_pattern(source_text: str, start: int, end: int) -> str:
    line_start = source_text.rfind("\n", 0, start) + 1
    line_end = source_text.find("\n", end)
    if line_end < 0:
        line_end = len(source_text)
    line = source_text[line_start:start] + " <ENTITY> " + source_text[end:line_end]
    normalized = " ".join(line.split())
    return normalized[:320]


def _proposal_id(document_id: str, start: int, end: int, entity_type: str) -> str:
    payload = f"{document_id}\0{start}\0{end}\0{entity_type}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _valid_position(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and value[0] < value[1]
    )


def _matrix_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    position = row["position"]
    return (_document_sort_key(str(row["document_id"])), position[0], position[1], row["type"])


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Any) -> None:
    values = rows if isinstance(rows, list) else []
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values),
        encoding="utf-8",
    )


def _write_review_csv(path: Path, rows: Any) -> None:
    fields = (
        "normalized_mention",
        "type",
        "status",
        "sources",
        "occurrence_count",
        "document_support",
        "missing_source_count",
        "context_patterns",
        "examples",
        "review_decision",
        "review_notes",
    )
    values = rows if isinstance(rows, list) else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False)
                    if isinstance(row.get(field), list | dict)
                    else row.get(field, "")
                    for field in fields
                }
            )
