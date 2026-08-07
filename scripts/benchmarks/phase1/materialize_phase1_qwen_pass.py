#!/usr/bin/env python
"""Materialize a stored Qwen extraction pass as an offset-safe Phase 1 source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from clingrounder.benchmarks.phase1.qwen_runner import (
    materialize_phase1_qwen_pass_source,
)
from clingrounder.benchmarks.phase1.round2 import load_phase1_round2_documents
from clingrounder.mining.io import load_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--raw-responses", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pass-id", default="recall")
    parser.add_argument("--max-window-characters", type=int, default=12_000)
    parser.add_argument("--window-overlap-characters", type=int, default=800)
    args = parser.parse_args()

    documents = load_phase1_round2_documents(
        load_documents(args.documents),
        expected_archive_sha256=args.expected_source_archive_sha256,
    )
    raw_records = tuple(
        json.loads(line)
        for line in Path(args.raw_responses).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    manifest = materialize_phase1_qwen_pass_source(
        documents,
        raw_records,
        args.output_dir,
        pass_id=args.pass_id,
        max_window_characters=args.max_window_characters,
        window_overlap_characters=args.window_overlap_characters,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
