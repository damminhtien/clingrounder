from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


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
