#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.alias_mining import (
    mine_vietnamese_alias_candidates,
    write_alias_mining_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine Vietnamese clinical alias candidates from input notes and full standards.",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--runtime-dictionary",
        default="data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl",
        help="Current runtime/controlled dictionary. Candidates already covered here are suppressed.",
    )
    parser.add_argument(
        "--standard-dictionary",
        action="append",
        default=[],
        help="Full standards dictionary JSONL. Repeat for TT06, RxNorm, LOCAL packs, etc.",
    )
    parser.add_argument("--output-dir", default="outputs/source_audit/alias_mining")
    parser.add_argument("--top-k-unknown", type=int, default=120)
    parser.add_argument("--top-k-abbreviations", type=int, default=80)
    args = parser.parse_args()

    standard_paths = args.standard_dictionary or [
        "data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl",
        "data/standards/rxnorm/processed/rxnorm_prescribable_06012026_concepts.jsonl",
    ]
    candidates = mine_vietnamese_alias_candidates(
        input_dir=args.input_dir,
        runtime_dictionary_path=args.runtime_dictionary,
        standard_dictionary_paths=standard_paths,
        top_k_unknown=args.top_k_unknown,
        top_k_abbreviations=args.top_k_abbreviations,
    )
    write_alias_mining_outputs(candidates, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "alias_candidates": str(Path(args.output_dir) / "alias_candidates.jsonl"),
                "markdown": str(Path(args.output_dir) / "alias_candidates.md"),
                "candidate_count": len(candidates),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
