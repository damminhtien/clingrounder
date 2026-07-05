#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.rxnorm_sources import (
    RXNORM_FULL_2026_06_01_SOURCE_ID,
    RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID,
    build_rxnorm_concept_rows,
    parse_rxnorm_rxnconso,
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
            "Expected locked file: RxNorm_full_prescribe_06012026.zip."
        ),
    )
    parser.add_argument(
        "--full-rxnorm",
        action="append",
        default=[],
        help="RxNorm Full Monthly Release ZIP/RXNCONSO.RRF fallback. Expected: RxNorm_full_06012026.zip.",
    )
    parser.add_argument("--output", required=True, help="Output RxNorm dictionary JSONL.")
    parser.add_argument("--manifest", help="Optional output import manifest JSON.")
    args = parser.parse_args()

    terms = []
    source_inputs: list[str] = []
    source_parse_counts: list[dict[str, object]] = []
    for path in args.prescribable_rxnorm:
        source_inputs.append(path)
        parsed = parse_rxnorm_rxnconso(path, source_id=RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID)
        _require_parsed(path, "prescribable_rxnorm", len(parsed))
        source_parse_counts.append({"path": path, "parser": "prescribable_rxnorm", "terms": len(parsed)})
        terms.extend(parsed)
    for path in args.full_rxnorm:
        source_inputs.append(path)
        parsed = parse_rxnorm_rxnconso(path, source_id=RXNORM_FULL_2026_06_01_SOURCE_ID)
        _require_parsed(path, "full_rxnorm", len(parsed))
        source_parse_counts.append({"path": path, "parser": "full_rxnorm", "terms": len(parsed)})
        terms.extend(parsed)
    if not terms:
        raise SystemExit("At least one --prescribable-rxnorm or --full-rxnorm source file is required.")

    rows = build_rxnorm_concept_rows(terms)
    write_rxnorm_concept_rows(args.output, rows)
    manifest = {
        "concepts": len(rows),
        "output": args.output,
        "source_policy": rxnorm_source_policy(),
        "source_parse_counts": source_parse_counts,
        "source_inputs": source_inputs,
    }
    if args.manifest:
        manifest = write_rxnorm_import_manifest(
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
        f"No RxNorm terms parsed from {path!r} with {parser_name}. "
        "Use RXNCONSO.RRF or a release ZIP containing RXNCONSO.RRF."
    )


if __name__ == "__main__":
    main()
