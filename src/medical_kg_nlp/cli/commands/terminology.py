"""Persistent terminology build and inspection commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology import (
    SQLiteTerminologyRepository,
    build_terminology_index,
    evaluate_terminology_queries,
    load_terminology_queries,
    write_alias_overlay_query_set,
    write_linked_proposal_query_set,
)

__all__ = ["benchmark_index", "build_index", "build_query_set", "inspect_index"]


def build_index(args: argparse.Namespace) -> int:
    """Build and print a content-addressed terminology manifest."""

    manifest = build_terminology_index(
        tuple(args.source),
        alias_overlay_paths=tuple(args.alias_overlay),
        output_path=args.output,
        cache_dir=args.cache_dir,
    )
    payload = manifest.to_json()
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.manifest_output is not None:
        output = Path(args.manifest_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


def inspect_index(args: argparse.Namespace) -> int:
    """Print index metadata and optional filtered lexical query results."""

    repository = SQLiteTerminologyRepository(
        args.index,
        expected_source_paths=tuple(args.source) if args.source else None,
        expected_alias_overlay_paths=(tuple(args.alias_overlay) if args.source else None),
    )
    payload: dict[str, object] = {"metadata": repository.metadata}
    if args.query:
        entity_type = EntityType(args.entity_type) if args.entity_type else None
        code_systems = (
            tuple(CodeSystem(value) for value in args.code_system)
            if args.code_system
            else None
        )
        payload["results"] = [
            {
                "concept_id": hit.entry.concept_id,
                "code": hit.entry.code,
                "code_system": hit.entry.code_system.value,
                "canonical_name": hit.entry.canonical_name,
                "semantic_type": hit.entry.semantic_type.value,
                "score": hit.score,
                "matched_alias": hit.matched_alias,
                "match_kind": hit.match_kind,
                "lexical_rank": hit.lexical_rank,
            }
            for hit in repository.search_scored(
                args.query,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=args.limit,
            )
        ]
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_query_set(args: argparse.Namespace) -> int:
    """Create a source-pinned retrieval query set without mixing train and held-out evidence."""

    if args.alias_overlay:
        if args.reference_alias_overlay:
            raise ValueError("Reference overlays apply only to linked proposal query sets")
        payload = write_alias_overlay_query_set(
            tuple(args.alias_overlay),
            args.output,
            args.manifest_output,
        )
    else:
        payload = write_linked_proposal_query_set(
            tuple(args.linked_proposal),
            args.output,
            args.manifest_output,
            reference_overlay_paths=tuple(args.reference_alias_overlay),
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def benchmark_index(args: argparse.Namespace) -> int:
    """Evaluate a fingerprint-validated index against neutral query JSONL."""

    repository = SQLiteTerminologyRepository(
        args.index,
        expected_source_paths=tuple(args.source),
        expected_alias_overlay_paths=tuple(args.alias_overlay),
    )
    try:
        report = evaluate_terminology_queries(
            repository,
            load_terminology_queries(args.queries),
            limit=args.limit,
        )
    finally:
        repository.close()
    payload = {
        **report,
        "index": str(Path(args.index)),
        "input_fingerprint": repository.metadata.get("input_fingerprint", ""),
        "queries": str(Path(args.queries)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # SCALING: detailed miss rows stay in the report file; agent polling remains compact.
    printed = payload if args.verbose else _benchmark_summary(payload, output)
    print(json.dumps(printed, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _benchmark_summary(payload: dict[str, object], output: Path) -> dict[str, object]:
    modes = payload.get("modes")
    compact_modes: dict[str, object] = {}
    if isinstance(modes, dict):
        for name, raw in modes.items():
            if isinstance(raw, dict):
                compact_modes[str(name)] = {
                    "metrics": raw.get("metrics", {}),
                    "latency_ms": raw.get("latency_ms", {}),
                }
    return {
        "schema_version": payload.get("schema_version"),
        "index": payload.get("index"),
        "queries": payload.get("queries"),
        "query_count": payload.get("query_count"),
        "unknown_expected_code_count": payload.get("unknown_expected_code_count"),
        "modes": compact_modes,
        "report": str(output),
    }
