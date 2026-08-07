#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from clingrounder.benchmarks.phase1.manual_gold import load_phase1_directory
from clingrounder.benchmarks.phase1.phase1_rule_registry import write_phase1_rule_registry
from clingrounder.benchmarks.phase1.phase1_selective_overlays import (
    compile_reviewed_candidate_registry,
    write_reviewed_candidate_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile reviewed exact-unique Phase 1 candidate mappings.",
    )
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument(
        "--dictionary",
        default="data/manual_gold/reviewed_candidate_concepts.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/manual_gold/reviewed_candidate_map.jsonl",
    )
    parser.add_argument(
        "--registry-output",
        default="data/manual_gold/derived/reviewed-candidate-registry.yaml",
    )
    parser.add_argument(
        "--audit-output",
        default="outputs/phase1/reviewed_candidates/audit.json",
    )
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    args = parser.parse_args()

    gold = load_phase1_directory(args.gold_dir)
    registry, audit = compile_reviewed_candidate_registry(
        gold,
        args.dictionary,
        split=args.split,
    )
    rows = write_reviewed_candidate_map(registry, args.output)
    registry_path = Path(args.registry_output)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    write_phase1_rule_registry(registry, registry_path)
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_payload = {
        **audit,
        "reviewed_candidate_map": str(Path(args.output)),
        "reviewed_candidate_count": len(rows),
        "rule_registry": str(registry_path),
    }
    audit_path.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
