#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.tt06_pdf import (
    build_tt06_manifest,
    extract_tt06_rows_from_pdf,
    extract_tt06_rows_from_tsv,
    run_pdftotext_tsv,
)
from medical_kg_nlp.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract TT 06/2026/TT-BYT ICD-10 appendix rows from PDF/TSV to JSONL.",
    )
    parser.add_argument("--pdf", required=True, help="Source 06-byt-kem.pdf.")
    parser.add_argument("--tsv", help="Optional pdftotext -tsv cache path.")
    parser.add_argument("--output", required=True, help="Output structured TT06 JSONL.")
    parser.add_argument("--manifest", help="Optional output manifest JSON.")
    parser.add_argument(
        "--reuse-tsv",
        action="store_true",
        help="Read existing --tsv instead of regenerating it from the PDF.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    tsv_path = Path(args.tsv) if args.tsv else output_path.with_suffix(".tsv")
    if args.reuse_tsv and tsv_path.exists():
        rows = extract_tt06_rows_from_tsv(tsv_path)
    elif args.tsv:
        run_pdftotext_tsv(args.pdf, tsv_path)
        rows = extract_tt06_rows_from_tsv(tsv_path)
    else:
        rows = extract_tt06_rows_from_pdf(args.pdf, tsv_path=tsv_path)

    write_jsonl(output_path, [row.to_json() for row in rows])
    manifest = build_tt06_manifest(
        rows=rows,
        source_pdf=args.pdf,
        output_jsonl=output_path,
        output_tsv=tsv_path,
    )
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
