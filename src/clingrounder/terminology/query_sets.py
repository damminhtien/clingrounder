"""Deterministic retrieval query sets derived from reviewed alias overlays."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.terminology.evaluation import TerminologyQuery
from clingrounder.utils.hashing import sha256_file, sha256_text
from clingrounder.utils.text import normalize_for_match

__all__ = [
    "build_alias_overlay_queries",
    "build_linked_proposal_queries",
    "write_alias_overlay_query_set",
    "write_linked_proposal_query_set",
]


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
    output = _write_queries(queries, output_path)

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


def build_linked_proposal_queries(
    proposal_paths: Sequence[str | Path],
    *,
    reference_overlay_paths: Sequence[str | Path] = (),
) -> tuple[TerminologyQuery, ...]:
    """Convert held-out concept-linked proposals into leakage-audited queries."""

    if not proposal_paths:
        raise ValueError("At least one linked proposal source is required")
    reference_aliases, reference_codes = _reference_knowledge(reference_overlay_paths)
    grouped_codes: dict[tuple[str, EntityType, CodeSystem], set[str]] = defaultdict(set)
    grouped_surfaces: dict[tuple[str, EntityType, CodeSystem], set[str]] = defaultdict(set)
    for raw_path in proposal_paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                normalized = _required_string(raw, "normalized_alias", path, line_number)
                if normalized != normalize_for_match(normalized):
                    raise ValueError(
                        f"{path}:{line_number}: normalized_alias violates lookup contract"
                    )
                entity_type = EntityType(
                    _required_string(raw, "semantic_type", path, line_number)
                )
                code_system = CodeSystem(
                    _required_string(raw, "code_system", path, line_number)
                )
                code = _required_string(raw, "code", path, line_number)
                key = (normalized, entity_type, code_system)
                grouped_codes[key].add(code)
                grouped_surfaces[key].update(
                    _proposal_surfaces(raw, path=path, line_number=line_number)
                )

    queries: list[TerminologyQuery] = []
    for normalized, entity_type, code_system in sorted(
        grouped_codes,
        key=lambda item: (item[1].value, item[2].value, item[0]),
    ):
        key = (normalized, entity_type, code_system)
        codes = tuple(sorted(grouped_codes[key]))
        mention = min(grouped_surfaces[key], key=lambda value: (value.casefold(), value))
        alias_seen = key in reference_aliases
        code_seen = any(
            (entity_type, code_system, code) in reference_codes for code in codes
        )
        identity = "\x1f".join((normalized, entity_type.value, code_system.value))
        queries.append(
            TerminologyQuery(
                query_id=f"linked-proposal:{sha256_text(identity)[:24]}",
                mention=mention,
                entity_type=entity_type,
                code_system=code_system,
                expected_codes=codes,
                slices=(
                    "alias_seen_in_reference" if alias_seen else "alias_unseen_in_reference",
                    "code_seen_in_reference" if code_seen else "code_unseen_in_reference",
                ),
            )
        )
    if not queries:
        raise ValueError("Linked proposals did not contain any retrieval queries")
    return tuple(queries)


def write_linked_proposal_query_set(
    proposal_paths: Sequence[str | Path],
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    reference_overlay_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Write held-out linked queries and pin both query and reference evidence."""

    queries = build_linked_proposal_queries(
        proposal_paths,
        reference_overlay_paths=reference_overlay_paths,
    )
    output = _write_queries(queries, output_path)
    manifest: dict[str, Any] = {
        "schema_version": "terminology-linked-proposal-query-set.v1",
        "query_count": len(queries),
        "ambiguous_query_count": sum(len(query.expected_codes) > 1 for query in queries),
        "entity_type_counts": _counts(query.entity_type.value for query in queries),
        "code_system_counts": _counts(query.code_system.value for query in queries),
        "slice_counts": _counts(value for query in queries for value in query.slices),
        "sources": _fingerprinted_paths(proposal_paths),
        "reference_alias_overlays": _fingerprinted_paths(reference_overlay_paths),
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


def _write_queries(
    queries: Sequence[TerminologyQuery],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for query in queries:
            row: dict[str, object] = {
                "code_system": query.code_system.value,
                "entity_type": query.entity_type.value,
                "expected_codes": list(query.expected_codes),
                "mention": query.mention,
                "query_id": query.query_id,
            }
            if query.slices:
                row["slices"] = list(query.slices)
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    return output


def _proposal_surfaces(
    raw: Mapping[str, object],
    *,
    path: Path,
    line_number: int,
) -> set[str]:
    variants = raw.get("surface_variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"{path}:{line_number}: surface_variants must be a non-empty array")
    surfaces: set[str] = set()
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError(f"{path}:{line_number}: invalid surface variant")
        surface = variant.get("surface")
        if not isinstance(surface, str) or not surface.strip():
            raise ValueError(f"{path}:{line_number}: surface must be non-empty")
        surfaces.add(surface.strip())
    return surfaces


def _reference_knowledge(
    overlay_paths: Sequence[str | Path],
) -> tuple[
    set[tuple[str, EntityType, CodeSystem]],
    set[tuple[EntityType, CodeSystem, str]],
]:
    aliases: set[tuple[str, EntityType, CodeSystem]] = set()
    codes: set[tuple[EntityType, CodeSystem, str]] = set()
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
                entity_type = EntityType(
                    _required_string(raw, "semantic_type", path, line_number)
                )
                code_system = CodeSystem(
                    _required_string(raw, "code_system", path, line_number)
                )
                code = _required_string(raw, "code", path, line_number)
                aliases.add((normalize_for_match(alias), entity_type, code_system))
                codes.add((entity_type, code_system, code))
    return aliases, codes


def _fingerprinted_paths(paths: Sequence[str | Path]) -> list[dict[str, str]]:
    return [
        {"path": str(Path(path)), "sha256": sha256_file(path)}
        for path in sorted(paths, key=lambda value: str(Path(value)))
    ]
