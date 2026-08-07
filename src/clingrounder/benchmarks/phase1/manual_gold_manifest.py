from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from clingrounder.utils.io import read_source_text


DEFAULT_CANDIDATE_POLICY = (
    "Use exact ICD-10 codes from TT06/2026 and active RxCUI values from the locked "
    "RxNorm 2026-07-06 release; leave [] when the mapping is ambiguous or unavailable."
)


def sync_manual_gold_manifest(
    gold_dir: str | Path,
    input_dir: str | Path,
    manifest_path: str | Path,
    *,
    status: str,
    reviewed_by: str,
    review_date: str,
    refresh_candidate_policy: bool = False,
) -> list[dict[str, Any]]:
    """Synchronize manifest coverage and counts without replacing existing review notes."""
    gold_root = Path(gold_dir)
    input_root = Path(input_dir)
    existing = _load_manifest(Path(manifest_path))
    output: list[dict[str, Any]] = []
    gold_paths = sorted(
        (path for path in gold_root.glob("*.json") if path.stem.isdigit()),
        key=lambda path: _document_sort_key(path.stem),
    )
    for gold_path in gold_paths:
        document_id = gold_path.stem
        entities = json.loads(gold_path.read_text(encoding="utf-8"))
        if not isinstance(entities, list):
            raise ValueError(f"{gold_path}: expected a JSON list.")
        row = dict(existing.get(document_id, {}))
        if not row:
            # New records explicitly state how proposal output was used so later readers do not
            # mistake model-assisted review for copied pseudo-gold.
            row = {
                "document_id": document_id,
                "status": status,
                "reviewed_by": reviewed_by,
                "review_date": review_date,
                "candidate_policy": DEFAULT_CANDIDATE_POLICY,
                "draft_policy": (
                    "Pipeline output was used only as a proposal seed; accepted entities were "
                    "reviewed against raw text and offsets."
                ),
                "guideline_notes": [
                    "Apply BTC medication full-SIG boundaries where the source provides them.",
                    "Exclude procedures and devices because Phase 1 has no matching type.",
                    "Keep repeated occurrences as separate span-level entities.",
                ],
            }
        row.update(
            {
                "document_id": document_id,
                "gold_file": str(gold_path),
                "source_file": str(input_root / f"{document_id}.txt"),
                "entity_count": len(entities),
            }
        )
        source_path = input_root / f"{document_id}.txt"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing raw source for manifest row {document_id}: {source_path}")
        _repair_review_candidate_positions(row, read_source_text(source_path))
        if refresh_candidate_policy:
            # Candidate details belong to generated standards-backed resources. Keeping code
            # inventories in prose made the manifest stale whenever a reviewed mapping changed.
            row["candidate_policy"] = DEFAULT_CANDIDATE_POLICY
        output.append(row)
    return output


def validate_manual_gold_manifest(
    gold_by_id: Mapping[str, list[dict[str, Any]]],
    manifest_path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        return [{"kind": "missing_review_manifest", "path": str(path)}]
    rows = _load_manifest(path)
    issues: list[dict[str, Any]] = []
    for document_id, entities in gold_by_id.items():
        row = rows.get(document_id)
        if row is None:
            issues.append(
                {
                    "kind": "missing_review_manifest_row",
                    "document_id": document_id,
                    "path": str(path),
                }
            )
            continue
        if row.get("entity_count") != len(entities):
            issues.append(
                {
                    "kind": "review_manifest_entity_count",
                    "document_id": document_id,
                    "path": str(path),
                    "expected": len(entities),
                    "actual": row.get("entity_count"),
                }
            )
        for field in ("status", "reviewed_by", "review_date", "candidate_policy", "draft_policy"):
            if not str(row.get(field, "")).strip():
                issues.append(
                    {
                        "kind": "review_manifest_required_field",
                        "document_id": document_id,
                        "path": str(path),
                        "field": field,
                    }
                )
        source_file = Path(str(row.get("source_file", "")))
        if not source_file.exists():
            issues.append(
                {
                    "kind": "review_manifest_source_file",
                    "document_id": document_id,
                    "path": str(source_file),
                }
            )
            continue
        source_text = read_source_text(source_file)
        review_candidates = row.get("review_candidates", [])
        if not isinstance(review_candidates, list):
            issues.append(
                {
                    "kind": "review_manifest_candidates_schema",
                    "document_id": document_id,
                }
            )
            continue
        for index, candidate in enumerate(review_candidates):
            if not isinstance(candidate, Mapping):
                issues.append(
                    {
                        "kind": "review_manifest_candidate_schema",
                        "document_id": document_id,
                        "index": index,
                    }
                )
                continue
            scope = str(candidate.get("scope", "entity")).strip()
            if scope not in {"entity", "candidate_mapping", "annotation_note"}:
                issues.append(
                    {
                        "kind": "review_manifest_candidate_scope",
                        "document_id": document_id,
                        "index": index,
                        "scope": scope,
                    }
                )
            position = candidate.get("position")
            if position is None:
                continue
            if not _valid_position(position):
                issues.append(
                    {
                        "kind": "review_manifest_candidate_position",
                        "document_id": document_id,
                        "index": index,
                        "position": position,
                    }
                )
                continue
            start, end = position
            text = str(candidate.get("text", ""))
            if (
                end > len(source_text)
                or source_text[start:end] != text
                or not _has_surface_boundaries(source_text, start, end, text)
            ):
                issues.append(
                    {
                        "kind": "review_manifest_candidate_offset",
                        "document_id": document_id,
                        "index": index,
                        "position": position,
                        "text": text,
                    }
                )
    for document_id in sorted(set(rows) - set(gold_by_id), key=_document_sort_key):
        issues.append(
            {
                "kind": "orphan_review_manifest_row",
                "document_id": document_id,
                "path": str(path),
            }
        )
    return issues


def write_manual_gold_manifest(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in materialized
        ),
        encoding="utf-8",
    )
    return len(materialized)


def _repair_review_candidate_positions(row: dict[str, Any], source_text: str) -> None:
    """Realign review evidence to raw text without inventing an approximate offset."""
    candidates = row.get("review_candidates")
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        text = str(candidate.get("text", ""))
        position = candidate.get("position")
        if not text or position is None:
            continue
        if _valid_position(position):
            start, end = position
            if (
                end <= len(source_text)
                and source_text[start:end] == text
                and _has_surface_boundaries(source_text, start, end, text)
            ):
                continue
            expected_start: int | None = start
        else:
            expected_start = None

        matches = _surface_matches(source_text, text)
        selected = _nearest_unambiguous_match(matches, expected_start)
        if selected is None:
            # Review candidates are negative/audit evidence, not gold entities. A null position is
            # safer than silently binding that evidence to the wrong occurrence after source edits.
            candidate["position"] = None
            continue
        start, end = selected
        candidate["text"] = source_text[start:end]
        candidate["position"] = [start, end]


def _surface_matches(source_text: str, text: str) -> list[tuple[int, int]]:
    exact = _literal_matches(source_text, text)
    # Human review notes commonly normalize capitalization or repeated whitespace. Capture the
    # exact raw surface once aligned so all persisted offsets remain executable specifications.
    tokens = text.split()
    if not tokens:
        return exact
    pattern = re.compile(r"\s+".join(re.escape(token) for token in tokens), re.IGNORECASE)
    relaxed = [
        (match.start(), match.end())
        for match in pattern.finditer(source_text)
        if _has_surface_boundaries(source_text, match.start(), match.end(), text)
    ]
    return sorted(set(exact) | set(relaxed))


def _literal_matches(source_text: str, text: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    start = source_text.find(text)
    while start >= 0:
        end = start + len(text)
        if _has_surface_boundaries(source_text, start, end, text):
            matches.append((start, end))
        start = source_text.find(text, start + 1)
    return matches


def _has_surface_boundaries(source_text: str, start: int, end: int, text: str) -> bool:
    if text[0].isalnum() and start > 0 and source_text[start - 1].isalnum():
        return False
    return not (text[-1].isalnum() and end < len(source_text) and source_text[end].isalnum())


def _nearest_unambiguous_match(
    matches: list[tuple[int, int]], expected_start: int | None
) -> tuple[int, int] | None:
    if not matches:
        return None
    if expected_start is None:
        return matches[0] if len(matches) == 1 else None
    ranked = sorted(matches, key=lambda span: (abs(span[0] - expected_start), span[0]))
    if len(ranked) > 1 and abs(ranked[0][0] - expected_start) == abs(
        ranked[1][0] - expected_start
    ):
        return None
    return ranked[0]


def _valid_position(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
        and 0 <= value[0] < value[1]
    )


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object.")
            document_id = str(row.get("document_id", "")).strip()
            if not document_id:
                raise ValueError(f"{path}:{line_number}: document_id is required.")
            if document_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate document_id {document_id!r}.")
            rows[document_id] = row
    return rows


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
