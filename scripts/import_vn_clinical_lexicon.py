#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.vn_clinical_lexicon import (
    VN_CLINICAL_LEXICON_SOURCE_ID,
    parse_vn_clinical_lexicon,
    write_vn_clinical_lexicon_manifest,
    write_vn_clinical_lexicon_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reviewed Vietnamese LOCAL clinical ConceptEntry rows from TSV/CSV curation files.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Reviewed Vietnamese clinical lexicon TSV/CSV. Can be repeated.",
    )
    parser.add_argument("--output", required=True, help="Output ConceptEntry JSONL.")
    parser.add_argument("--manifest", help="Optional output import manifest JSON.")
    parser.add_argument("--source-id", default=VN_CLINICAL_LEXICON_SOURCE_ID)
    args = parser.parse_args()

    rows = []
    warnings = []
    for path in args.input:
        parsed_rows, parsed_warnings = parse_vn_clinical_lexicon(path, source_id=args.source_id)
        rows.extend(parsed_rows)
        warnings.extend({"path": path, **warning} for warning in parsed_warnings)
    rows = sorted(rows, key=lambda row: str(row["concept_id"]))
    write_vn_clinical_lexicon_rows(args.output, rows)
    manifest = write_vn_clinical_lexicon_manifest(
        args.manifest or str(Path(args.output).with_suffix(".manifest.json")),
        rows=rows,
        source_inputs=args.input,
        parse_warnings=warnings,
        source_id=args.source_id,
    )
    manifest["output"] = args.output
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
