#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.rxnorm_sources import (
    RXNORM_FULL_2026_06_01_SOURCE_ID,
    RXNORM_ENRICHMENT_TTYS,
    RXNORM_FULL_FALLBACK_TTYS,
    RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID,
    build_rxnorm_concept_rows,
    parse_rxnorm_rxnconso,
    parse_rxnorm_rxnrel,
    parse_rxnorm_rxnsat,
    profile_rxnorm_release,
    resolve_rxnorm_archive_member_root,
    rxnorm_source_policy,
    write_rxnorm_concept_rows,
    write_rxnorm_import_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RxNorm ConceptEntry JSONL rows from local NLM RXNCONSO.RRF releases.",
    )
    parser.add_argument(
        "--prescribable-rxnorm",
        action="append",
        default=[],
        help=(
            "RxNorm Current Prescribable Content ZIP/RXNCONSO.RRF. "
            "A bundled full archive can be used when it contains prescribe/rrf."
        ),
    )
    parser.add_argument(
        "--full-rxnorm",
        action="append",
        default=[],
        help="RxNorm Full Monthly Release ZIP/RXNCONSO.RRF fallback.",
    )
    parser.add_argument(
        "--prescribable-source-id",
        default=RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID,
        help="Provenance source id assigned to prescribable rows.",
    )
    parser.add_argument(
        "--full-source-id",
        default=RXNORM_FULL_2026_06_01_SOURCE_ID,
        help="Provenance source id assigned to full-release rows.",
    )
    parser.add_argument("--release-date", default="2026-06-01", help="ISO release date for the import manifest.")
    parser.add_argument("--primary-file", default="RxNorm_full_prescribe_06012026.zip")
    parser.add_argument("--fallback-file", default="RxNorm_full_06012026.zip")
    parser.add_argument("--output", required=True, help="Output RxNorm dictionary JSONL.")
    parser.add_argument("--manifest", help="Optional output import manifest JSON.")
    args = parser.parse_args()

    terms = []
    enrichment_terms = []
    relations = []
    attributes = []
    source_inputs: list[str] = []
    source_parse_counts: list[dict[str, object]] = []
    release_profiles: list[dict[str, object]] = []
    for path in args.prescribable_rxnorm:
        archive_member_root = resolve_rxnorm_archive_member_root(path, content="prescribable")
        source_inputs.append(path)
        parsed = parse_rxnorm_rxnconso(
            path,
            source_id=args.prescribable_source_id,
            archive_member_root=archive_member_root,
        )
        parsed_enrichment_terms = parse_rxnorm_rxnconso(
            path,
            source_id=args.prescribable_source_id,
            allowed_ttys=RXNORM_ENRICHMENT_TTYS,
            archive_member_root=archive_member_root,
        )
        parsed_relations = parse_rxnorm_rxnrel(
            path,
            source_id=args.prescribable_source_id,
            archive_member_root=archive_member_root,
        )
        parsed_attributes = parse_rxnorm_rxnsat(
            path,
            source_id=args.prescribable_source_id,
            archive_member_root=archive_member_root,
        )
        _require_parsed(path, "prescribable_rxnorm", len(parsed))
        source_parse_counts.append(
            {
                "path": path,
                "parser": "prescribable_rxnorm",
                "terms": len(parsed),
                "enrichment_terms": len(parsed_enrichment_terms),
                "relations": len(parsed_relations),
                "attributes": len(parsed_attributes),
                "archive_member_root": archive_member_root,
            }
        )
        release_profiles.append(profile_rxnorm_release(path, archive_member_root=archive_member_root))
        terms.extend(parsed)
        enrichment_terms.extend(parsed_enrichment_terms)
        relations.extend(parsed_relations)
        attributes.extend(parsed_attributes)
    for path in args.full_rxnorm:
        archive_member_root = resolve_rxnorm_archive_member_root(path, content="full")
        source_inputs.append(path)
        parsed = parse_rxnorm_rxnconso(
            path,
            source_id=args.full_source_id,
            allowed_ttys=RXNORM_FULL_FALLBACK_TTYS,
            archive_member_root=archive_member_root,
        )
        parsed_enrichment_terms = parse_rxnorm_rxnconso(
            path,
            source_id=args.full_source_id,
            allowed_ttys=RXNORM_ENRICHMENT_TTYS,
            archive_member_root=archive_member_root,
        )
        parsed_relations = parse_rxnorm_rxnrel(path, source_id=args.full_source_id, archive_member_root=archive_member_root)
        parsed_attributes = parse_rxnorm_rxnsat(path, source_id=args.full_source_id, archive_member_root=archive_member_root)
        _require_parsed(path, "full_rxnorm", len(parsed))
        source_parse_counts.append(
            {
                "path": path,
                "parser": "full_rxnorm",
                "terms": len(parsed),
                "enrichment_terms": len(parsed_enrichment_terms),
                "relations": len(parsed_relations),
                "attributes": len(parsed_attributes),
                "archive_member_root": archive_member_root,
            }
        )
        release_profiles.append(
            profile_rxnorm_release(
                path,
                allowed_ttys=RXNORM_FULL_FALLBACK_TTYS,
                archive_member_root=archive_member_root,
            )
        )
        terms.extend(parsed)
        enrichment_terms.extend(parsed_enrichment_terms)
        relations.extend(parsed_relations)
        attributes.extend(parsed_attributes)
    if not terms:
        raise SystemExit("At least one --prescribable-rxnorm or --full-rxnorm source file is required.")

    rows = build_rxnorm_concept_rows(
        terms,
        enrichment_terms=enrichment_terms,
        relations=relations,
        attributes=attributes,
        source_priority=(args.prescribable_source_id, args.full_source_id),
    )
    write_rxnorm_concept_rows(args.output, rows)
    source_policy = rxnorm_source_policy(
        primary_source_id=args.prescribable_source_id,
        fallback_source_id=args.full_source_id,
        release_date=args.release_date,
        primary_file=args.primary_file,
        fallback_file=args.fallback_file,
    )
    manifest = {
        "concepts": len(rows),
        "output": args.output,
        "source_policy": source_policy,
        "source_parse_counts": source_parse_counts,
        "source_inputs": source_inputs,
        "release_profiles": release_profiles,
    }
    if args.manifest:
        manifest = write_rxnorm_import_manifest(
            args.manifest,
            rows=rows,
            source_inputs=source_inputs,
            source_parse_counts=source_parse_counts,
            release_profiles=release_profiles,
            source_policy=source_policy,
        )
        manifest["output"] = args.output
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def _require_parsed(path: str, parser_name: str, count: int) -> None:
    if count > 0:
        return
    raise SystemExit(
        f"No RxNorm terms parsed from {path!r} with {parser_name}. "
        "Use RXNCONSO.RRF or a release ZIP containing RXNCONSO.RRF."
    )


if __name__ == "__main__":
    main()
