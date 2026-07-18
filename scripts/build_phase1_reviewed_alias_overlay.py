#!/usr/bin/env python
"""Build the opt-in normalization overlay from reviewed Phase 1 mappings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.benchmarks.phase1.reviewed_alias_overlay import (
    compile_reviewed_candidate_aliases,
    reviewed_alias_memory_rows,
)
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.terminology import SQLiteTerminologyRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile reviewed train mappings into a strict normalization overlay. "
            "Recognition remains unchanged."
        )
    )
    parser.add_argument("--map", required=True, dest="map_path")
    parser.add_argument("--index", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--base-alias-overlay", action="append", default=[])
    parser.add_argument("--overlay-output", required=True)
    parser.add_argument("--recognition-output", required=True)
    parser.add_argument("--memory-output", required=True)
    parser.add_argument("--decisions-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    sources = tuple(args.source)
    base_overlays = tuple(args.base_alias_overlay)
    repository = SQLiteTerminologyRepository(
        args.index,
        expected_source_paths=sources,
        expected_alias_overlay_paths=base_overlays,
    )
    metadata = dict(repository.metadata)
    try:
        result, map_sha256 = compile_reviewed_candidate_aliases(args.map_path, repository)
    finally:
        repository.close()

    overlay_sha256 = write_jsonl(args.overlay_output, result.alias_overlays)
    recognition_sha256 = write_jsonl(args.recognition_output, result.recognition_concepts)
    memory_sha256 = write_jsonl(args.memory_output, reviewed_alias_memory_rows(result))
    decisions_sha256 = write_jsonl(args.decisions_output, result.decisions)
    report = {
        **result.report,
        "inputs": {
            "reviewed_candidate_map": str(Path(args.map_path)),
            "reviewed_candidate_map_sha256": map_sha256,
            "terminology_index": str(Path(args.index)),
            "terminology_input_fingerprint": metadata.get("input_fingerprint", ""),
            "canonical_sources": [str(Path(path)) for path in sources],
            "base_alias_overlays": [str(Path(path)) for path in base_overlays],
        },
        "outputs": {
            "alias_overlay": str(Path(args.overlay_output)),
            "alias_overlay_sha256": overlay_sha256,
            "recognition_dictionary": str(Path(args.recognition_output)),
            "recognition_dictionary_sha256": recognition_sha256,
            "reviewed_memory": str(Path(args.memory_output)),
            "reviewed_memory_sha256": memory_sha256,
            "decisions": str(Path(args.decisions_output)),
            "decisions_sha256": decisions_sha256,
        },
        "runtime_contract": {
            "recognition_enabled": False,
            "normalization_overlay_only": True,
            "document_specific_fields_present": False,
            "reviewed_memory_terminal": True,
        },
    }
    write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
