"""Persistent terminology build and inspection commands."""

from __future__ import annotations

import argparse
import json

from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology import SQLiteTerminologyRepository, build_terminology_index

__all__ = ["build_index", "inspect_index"]


def build_index(args: argparse.Namespace) -> int:
    """Build and print a content-addressed terminology manifest."""

    manifest = build_terminology_index(
        tuple(args.source),
        alias_overlay_paths=tuple(args.alias_overlay),
        output_path=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(manifest.to_json(), ensure_ascii=False, indent=2, sort_keys=True))
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
                "concept_id": entry.concept_id,
                "code": entry.code,
                "code_system": entry.code_system.value,
                "canonical_name": entry.canonical_name,
                "semantic_type": entry.semantic_type.value,
            }
            for entry in repository.search(
                args.query,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=args.limit,
            )
        ]
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
