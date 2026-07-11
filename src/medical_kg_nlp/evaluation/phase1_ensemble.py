from __future__ import annotations

from collections import Counter
from itertools import product
import json
from pathlib import Path
from typing import Any, Literal
import zipfile

from medical_kg_nlp.evaluation.manual_gold import load_phase1_directory, manual_gold_split
from medical_kg_nlp.evaluation.phase1 import score_phase1_documents


PHASE1_ENTITY_TYPE_ORDER = (
    "CHẨN_ĐOÁN",
    "THUỐC",
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
)
Phase1EnsembleSource = Literal[
    "primary",
    "secondary",
    "union",
    "intersection",
    "primary_preferred_union",
    "secondary_preferred_union",
]


def load_phase1_output_source(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load a Phase 1 directory or ZIP with JSON files at root or under output/."""
    source = Path(path)
    if source.is_dir():
        return load_phase1_directory(source)
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError(f"Phase 1 source must be a directory or ZIP file: {source}")

    rows: dict[str, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(source) as archive:
        json_names = sorted(name for name in archive.namelist() if name.endswith(".json"))
        for name in json_names:
            document_id = Path(name).stem
            if not document_id.isdigit():
                continue
            if document_id in rows:
                raise ValueError(f"Duplicate document {document_id}.json in {source}")
            payload = json.loads(archive.read(name).decode("utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"{source}:{name}: expected a JSON list.")
            rows[document_id] = payload
    return rows


def merge_phase1_outputs(
    primary_by_doc: dict[str, list[dict[str, Any]]],
    secondary_by_doc: dict[str, list[dict[str, Any]]],
    source_by_type: dict[str, Phase1EnsembleSource],
    *,
    document_ids: list[str] | None = None,
    empty_assertions_and_candidates: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Merge two flat Phase 1 outputs without changing text, type, or offsets."""
    missing_types = set(PHASE1_ENTITY_TYPE_ORDER) - set(source_by_type)
    if missing_types:
        raise ValueError(f"Missing source strategy for entity types: {sorted(missing_types)}")
    invalid_types = set(source_by_type) - set(PHASE1_ENTITY_TYPE_ORDER)
    if invalid_types:
        raise ValueError(f"Unknown Phase 1 entity types: {sorted(invalid_types)}")

    ids = document_ids or sorted(set(primary_by_doc) | set(secondary_by_doc), key=_document_sort_key)
    merged_by_doc: dict[str, list[dict[str, Any]]] = {}
    for document_id in ids:
        primary_rows = primary_by_doc.get(document_id, [])
        secondary_rows = secondary_by_doc.get(document_id, [])
        merged: list[dict[str, Any]] = []
        for entity_type in PHASE1_ENTITY_TYPE_ORDER:
            primary_typed = [row for row in primary_rows if row.get("type") == entity_type]
            secondary_typed = [row for row in secondary_rows if row.get("type") == entity_type]
            strategy = source_by_type[entity_type]
            if strategy == "primary":
                selected = _deduplicate_rows(primary_typed)
            elif strategy == "secondary":
                selected = _deduplicate_rows(secondary_typed)
            elif strategy == "union":
                selected = _deduplicate_rows(primary_typed + secondary_typed)
            elif strategy == "intersection":
                secondary_keys = {_entity_key(row) for row in secondary_typed}
                selected = [row for row in primary_typed if _entity_key(row) in secondary_keys]
            elif strategy == "primary_preferred_union":
                selected = _preferred_nonoverlapping_union(primary_typed, secondary_typed)
            elif strategy == "secondary_preferred_union":
                selected = _preferred_nonoverlapping_union(secondary_typed, primary_typed)
            else:
                raise ValueError(f"Unsupported source strategy: {strategy}")
            merged.extend(
                _copy_entity_only_row(row) if empty_assertions_and_candidates else dict(row)
                for row in selected
            )
        merged_by_doc[document_id] = sorted(merged, key=_entity_sort_key)
    return merged_by_doc


def rank_phase1_source_strategies(
    gold_by_doc: dict[str, list[dict[str, Any]]],
    primary_by_doc: dict[str, list[dict[str, Any]]],
    secondary_by_doc: dict[str, list[dict[str, Any]]],
    *,
    choices: tuple[Phase1EnsembleSource, ...] = ("primary", "secondary"),
) -> list[dict[str, Any]]:
    """Rank source-per-type strategies using train only; holdout is diagnostic."""
    if not choices:
        raise ValueError("At least one ensemble source choice is required.")
    document_ids = sorted(gold_by_doc, key=_document_sort_key)
    split_ids = {
        "all": document_ids,
        "train": [doc_id for doc_id in document_ids if manual_gold_split(doc_id) == "train"],
        "holdout": [doc_id for doc_id in document_ids if manual_gold_split(doc_id) == "holdout"],
    }
    ranked: list[dict[str, Any]] = []
    for combination in product(choices, repeat=len(PHASE1_ENTITY_TYPE_ORDER)):
        source_by_type = dict(zip(PHASE1_ENTITY_TYPE_ORDER, combination, strict=True))
        predictions = merge_phase1_outputs(
            primary_by_doc,
            secondary_by_doc,
            source_by_type,
            document_ids=document_ids,
        )
        split_reports: dict[str, Any] = {}
        for split_name, ids in split_ids.items():
            gold = {doc_id: gold_by_doc[doc_id] for doc_id in ids}
            pred = {doc_id: predictions.get(doc_id, []) for doc_id in ids}
            metrics, errors = score_phase1_documents(gold, pred)
            split_reports[split_name] = {
                "metrics": metrics,
                "error_counts": dict(sorted(Counter(row["error_type"] for row in errors).items())),
            }
        ranked.append({"source_by_type": source_by_type, "splits": split_reports})

    return sorted(
        ranked,
        key=lambda row: (
            row["splits"]["train"]["metrics"]["score"],
            row["splits"]["train"]["metrics"]["text_score"],
        ),
        reverse=True,
    )


def expand_repeated_phase1_mentions(
    rows_by_doc: dict[str, list[dict[str, Any]]],
    source_text_by_doc: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Recover repeated exact mentions lost by first-occurrence-only post-processing."""
    expanded_by_doc: dict[str, list[dict[str, Any]]] = {}
    for document_id, rows in rows_by_doc.items():
        source_text = source_text_by_doc.get(document_id)
        if source_text is None:
            expanded_by_doc[document_id] = _deduplicate_rows(rows)
            continue
        expanded: list[dict[str, Any]] = []
        for row in _deduplicate_rows(rows):
            text = row.get("text")
            position = row.get("position")
            if (
                not isinstance(text, str)
                or not text
                or not isinstance(position, list)
                or len(position) != 2
                or not all(isinstance(value, int) for value in position)
            ):
                expanded.append(row)
                continue
            start, end = position
            if start < 0 or end > len(source_text) or source_text[start:end] != text:
                expanded.append(row)
                continue
            line_start = source_text.rfind("\n", 0, start) + 1
            line_end = source_text.find("\n", end)
            if line_end < 0:
                line_end = len(source_text)
            cursor = line_start
            while cursor < line_end:
                occurrence = source_text.find(text, cursor, line_end)
                if occurrence < 0:
                    break
                repeated = dict(row)
                repeated["position"] = [occurrence, occurrence + len(text)]
                expanded.append(repeated)
                cursor = occurrence + len(text)
        expanded_by_doc[document_id] = sorted(_deduplicate_rows(expanded), key=_entity_sort_key)
    return expanded_by_doc


def _copy_entity_only_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": row.get("text"),
        "type": row.get("type"),
        "assertions": [],
        "candidates": [],
        "position": list(row.get("position", [])),
    }


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for row in rows:
        key = _entity_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)
    return deduplicated


def _preferred_nonoverlapping_union(
    preferred_rows: list[dict[str, Any]],
    additional_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    preferred = _deduplicate_rows(preferred_rows)
    selected = list(preferred)
    for row in _deduplicate_rows(additional_rows):
        if any(_rows_overlap(row, existing) for existing in preferred):
            continue
        selected.append(row)
    return _deduplicate_rows(selected)


def _rows_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    _, _, left_start, left_end = _entity_key(left)
    _, _, right_start, right_end = _entity_key(right)
    if min(left_start, left_end, right_start, right_end) < 0:
        return False
    return left_start < right_end and right_start < left_end


def _entity_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    position = row.get("position")
    if not isinstance(position, list) or len(position) != 2:
        return (str(row.get("type", "")), str(row.get("text", "")), -1, -1)
    return (
        str(row.get("type", "")),
        str(row.get("text", "")),
        int(position[0]),
        int(position[1]),
    )


def _entity_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    _, text, start, end = _entity_key(row)
    return (start, end, str(row.get("type", "")), text)


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
