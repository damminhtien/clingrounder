#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import load_phase1_output_source
from medical_kg_nlp.benchmarks.phase1.phase1_proposals import (
    build_phase1_proposal_matrix,
    write_phase1_proposal_matrix,
)
from medical_kg_nlp.utils.io import read_source_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Align independent Phase 1 entity proposal sources.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named Phase 1 output directory or ZIP; repeat for each independent source.",
    )
    parser.add_argument("--input-dir", default="data/raw/input")
    parser.add_argument("--output-dir", default="outputs/phase1/proposal_matrix")
    args = parser.parse_args()

    source_paths = _parse_sources(args.source, parser)
    sources = {name: load_phase1_output_source(path) for name, path in source_paths.items()}
    source_text_by_doc = {
        path.stem: read_source_text(path) for path in Path(args.input_dir).glob("*.txt")
    }
    report = build_phase1_proposal_matrix(sources, source_text_by_doc)
    write_phase1_proposal_matrix(report, args.output_dir)
    print(
        json.dumps(
            {"output_dir": args.output_dir, **report["summary"]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _parse_sources(values: list[str], parser: argparse.ArgumentParser) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            parser.error(f"Invalid --source {value!r}; expected NAME=PATH.")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in sources:
            parser.error(f"Duplicate or empty source name {name!r}.")
        path = Path(raw_path)
        if not path.exists():
            parser.error(f"Source path does not exist: {path}")
        sources[name] = path
    if len(sources) < 2:
        parser.error("At least two --source arguments are required.")
    return sources


if __name__ == "__main__":
    main()
