#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from medical_kg_nlp.benchmarks.phase1.manual_gold_candidates import (
    build_manual_gold_candidate_dictionary,
    write_manual_gold_candidate_dictionary,
)


DEFAULT_SOURCES = (
    "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl",
    "data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl",
    "data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a compact standards-backed dictionary for candidate codes used in "
            "Phase 1 manual gold."
        )
    )
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Dictionary source in priority order; repeat for multiple sources.",
    )
    parser.add_argument(
        "--output",
        default="data/manual_gold/reviewed_candidate_concepts.jsonl",
    )
    parser.add_argument(
        "--audit-output",
        default="outputs/manual_gold/candidate_dictionary_audit.json",
    )
    args = parser.parse_args()

    rows, audit = build_manual_gold_candidate_dictionary(
        args.gold_dir,
        args.sources or DEFAULT_SOURCES,
    )
    if audit["issue_count"]:
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(1)

    write_manual_gold_candidate_dictionary(rows, args.output)
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
