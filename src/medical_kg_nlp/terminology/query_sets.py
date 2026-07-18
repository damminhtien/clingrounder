"""Deterministic retrieval query sets derived from reviewed alias overlays."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.evaluation import TerminologyQuery
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["build_alias_overlay_queries", "write_alias_overlay_query_set"]


def build_alias_overlay_queries(
    overlay_paths: Sequence[str | Path],
) -> tuple[TerminologyQuery, ...]:
    """Convert strict, linked alias overlays into task-neutral retrieval queries.

    The grouping key mirrors the terminology index collision scope. One surface may
    legitimately exist in different semantic spaces, while conflicting codes inside
    the same entity-type/code-system space become one query with multiple accepted
    codes.
    """

    if not overlay_paths:
        raise ValueError("At least one alias overlay is required")

    grouped_codes: dict[tuple[str, EntityType, CodeSystem], set[str]] = defaultdict(set)
    grouped_surfaces: dict[tuple[str, EntityType, CodeSystem], set[str]] = defaultdict(set)
    for raw_path in overlay_paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                alias = _required_string(raw, "alias", path, line_number)
                code = _required_string(raw, "code", path, line_number)
                normalized = normalize_for_match(alias)
                if not normalized:
                    raise ValueError(f"{path}:{line_number}: alias normalizes to empty text")
                entity_type = EntityType(
                    _required_string(raw, "semantic_type", path, line_number)
                )
                code_system = CodeSystem(
                    _required_string(raw, "code_system", path, line_number)
                )
                if code_system == CodeSystem.NONE:
                    raise ValueError(
                        f"{path}:{line_number}: linked aliases cannot use code system NONE"
                    )
                group_key = (normalized, entity_type, code_system)
                grouped_codes[group_key].add(code)
                grouped_surfaces[group_key].add(alias)

    queries: list[TerminologyQuery] = []
    for normalized, entity_type, code_system in sorted(
        grouped_codes,
        key=lambda item: (item[1].value, item[2].value, item[0]),
    ):
        query_key = (normalized, entity_type, code_system)
        # INVARIANT: display spelling is deterministic and does not alter lookup text.
        mention = min(
            grouped_surfaces[query_key],
            key=lambda value: (value.casefold(), value),
        )
        identity = "\x1f".join((normalized, entity_type.value, code_system.value))
        queries.append(
            TerminologyQuery(
                query_id=f"alias-overlay:{sha256_text(identity)[:24]}",
                mention=mention,
                entity_type=entity_type,
                code_system=code_system,
                expected_codes=tuple(sorted(grouped_codes[query_key])),
            )
        )
    if not queries:
        raise ValueError("Alias overlays did not contain any linked aliases")
    return tuple(queries)


def write_alias_overlay_query_set(
    overlay_paths: Sequence[str | Path],
    output_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Write deterministic JSONL plus a content-pinned reproducibility manifest."""

    queries = build_alias_overlay_queries(overlay_paths)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(
                json.dumps(
                    {
                        "code_system": query.code_system.value,
                        "entity_type": query.entity_type.value,
                        "expected_codes": list(query.expected_codes),
                        "mention": query.mention,
                        "query_id": query.query_id,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    sources = [
        {"path": str(Path(path)), "sha256": sha256_file(path)}
        for path in sorted(overlay_paths, key=lambda value: str(Path(value)))
    ]
    manifest: dict[str, Any] = {
        "schema_version": "terminology-alias-query-set.v1",
        "query_count": len(queries),
        "ambiguous_query_count": sum(len(query.expected_codes) > 1 for query in queries),
        "entity_type_counts": _counts(query.entity_type.value for query in queries),
        "code_system_counts": _counts(query.code_system.value for query in queries),
        "sources": sources,
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _required_string(
    raw: Mapping[str, object],
    field: str,
    path: Path,
    line_number: int,
) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: {field} must be a non-empty string")
    return value.strip()


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
