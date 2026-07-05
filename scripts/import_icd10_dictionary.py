#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.icd10_sources import (
    build_icd10_concept_rows,
    icd10_source_policy,
    load_icd10_vietnamese_overlays,
    parse_cdc_icd10cm_descriptions,
    parse_cdc_icd10cm_tabular_xml,
    parse_icd10_vn_tt06_table,
    parse_who_icd10_claml,
    write_icd10_concept_rows,
    write_icd10_import_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build ICD-10 ConceptEntry JSONL rows from local TT 06/2026/TT-BYT extracts "
            "plus optional WHO/CDC reference files."
        )
    )
    parser.add_argument(
        "--icd10-vn-tt06",
        action="append",
        default=[],
        help=(
            "Reviewed structured extract of TT 06/2026/TT-BYT ICD-10 appendix "
            "(JSON/JSONL/CSV/TSV or ZIP containing those files). Can be repeated."
        ),
    )
    parser.add_argument(
        "--who-claml",
        action="append",
        default=[],
        help="WHO ICD-10 ClaML XML file or ZIP containing XML. Reference/fallback only.",
    )
    parser.add_argument(
        "--cdc-descriptions",
        action="append",
        default=[],
        help="CDC ICD-10-CM code-description TXT file or ZIP containing TXT/CSV. Reference only.",
    )
    parser.add_argument(
        "--cdc-xml",
        action="append",
        default=[],
        help="CDC ICD-10-CM tabular XML file or ZIP containing tabular XML. Reference only.",
    )
    parser.add_argument(
        "--vietnamese-aliases",
        action="append",
        default=[],
        help=(
            "Curated Vietnamese alias JSONL. Supports existing target_concept_id rows or rows with "
            "code/aliases/official_name_vi. Can be repeated."
        ),
    )
    parser.add_argument("--output", required=True, help="Output ICD-10 dictionary JSONL.")
    parser.add_argument("--manifest", help="Optional output import manifest JSON.")
    parser.add_argument(
        "--allow-non-vietnamese-primary",
        action="store_true",
        help=(
            "Allow building from WHO/CDC without a TT 06/2026 source. Use only for tests or "
            "reference experiments, not Phase 1 primary dictionaries."
        ),
    )
    args = parser.parse_args()

    source_concepts = []
    source_inputs: list[str] = []
    source_parse_counts: list[dict[str, object]] = []
    for path in args.icd10_vn_tt06:
        source_inputs.append(path)
        parsed = parse_icd10_vn_tt06_table(path)
        _require_parsed(path, "icd10_vn_tt06", len(parsed))
        source_parse_counts.append({"path": path, "parser": "icd10_vn_tt06", "concepts": len(parsed)})
        source_concepts.extend(parsed)
    for path in args.who_claml:
        source_inputs.append(path)
        parsed = parse_who_icd10_claml(path)
        _require_parsed(path, "who_claml", len(parsed))
        source_parse_counts.append({"path": path, "parser": "who_claml", "concepts": len(parsed)})
        source_concepts.extend(parsed)
    for path in args.cdc_descriptions:
        source_inputs.append(path)
        parsed = parse_cdc_icd10cm_descriptions(path)
        _require_parsed(path, "cdc_descriptions", len(parsed))
        source_parse_counts.append({"path": path, "parser": "cdc_descriptions", "concepts": len(parsed)})
        source_concepts.extend(parsed)
    for path in args.cdc_xml:
        source_inputs.append(path)
        parsed = parse_cdc_icd10cm_tabular_xml(path)
        _require_parsed(path, "cdc_xml", len(parsed))
        source_parse_counts.append({"path": path, "parser": "cdc_xml", "concepts": len(parsed)})
        source_concepts.extend(parsed)
    if not source_concepts:
        raise SystemExit(
            "At least one --icd10-vn-tt06, --who-claml, --cdc-descriptions, or --cdc-xml "
            "source file is required."
        )
    if not args.icd10_vn_tt06 and not args.allow_non_vietnamese_primary:
        raise SystemExit(
            "ICD-10 Vietnamese primary source is required for Phase 1 dictionaries. "
            "Pass --icd10-vn-tt06 with a reviewed TT 06/2026/TT-BYT extract, or use "
            "--allow-non-vietnamese-primary for reference-only experiments."
        )

    overlays = []
    for path in args.vietnamese_aliases:
        source_inputs.append(path)
        overlays.extend(load_icd10_vietnamese_overlays(path))

    rows = build_icd10_concept_rows(source_concepts, overlays)
    write_icd10_concept_rows(args.output, rows)
    manifest = {
        "concepts": len(rows),
        "output": args.output,
        "source_policy": icd10_source_policy(),
        "source_parse_counts": source_parse_counts,
        "source_inputs": source_inputs,
    }
    if args.manifest:
        manifest = write_icd10_import_manifest(
            args.manifest,
            rows=rows,
            source_inputs=source_inputs,
            source_parse_counts=source_parse_counts,
        )
        manifest["output"] = args.output
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def _require_parsed(path: str, parser_name: str, count: int) -> None:
    if count > 0:
        return
    raise SystemExit(
        f"No ICD-10 concepts parsed from {path!r} with {parser_name}. "
        "Use WHO ClaML XML, CDC tabular XML, or a plain TXT/CSV descriptions file."
    )


if __name__ == "__main__":
    main()
