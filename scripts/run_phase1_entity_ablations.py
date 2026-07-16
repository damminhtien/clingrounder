#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.benchmarks.phase1.phase1_entity_ablation import (
    Phase1EntityAblationConfig,
    run_phase1_entity_ablations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated Phase 1 entity WER/source/boundary ablations."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--input-dir", default="data/raw/input")
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument(
        "--split-manifest",
        default="data/manual_gold/holdout_manifest.json",
    )
    parser.add_argument(
        "--annotation-policy",
        default="data/manual_gold/compiled/phase1_annotation_policy.yaml",
    )
    parser.add_argument(
        "--dictionary",
        action="append",
        default=[],
        help="Dictionary JSONL used for strict validation; repeat to load a union.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Ordered source-lineage stage for WER attribution.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/evaluation/phase1_entity_ablations",
    )
    parser.add_argument("--journal-dir", default="outputs/loops/journal")
    parser.add_argument("--public-wer", type=float)
    parser.add_argument("--minimum-boundary-document-support", type=int, default=2)
    args = parser.parse_args()

    dictionary_paths = tuple(Path(value) for value in args.dictionary) or (
        Path("data/standards/phase1_seed_tt06_controlled_concepts.jsonl"),
        Path("data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl"),
    )
    config = Phase1EntityAblationConfig(
        base=Path(args.base),
        expected_base_sha256=args.expected_base_sha256,
        input_dir=Path(args.input_dir),
        gold_dir=Path(args.gold_dir),
        split_manifest=Path(args.split_manifest),
        annotation_policy=Path(args.annotation_policy),
        dictionary_paths=dictionary_paths,
        source_stages=tuple(_parse_stage(value, parser) for value in args.stage),
        output_root=Path(args.output_root),
        journal_dir=Path(args.journal_dir),
        public_wer=args.public_wer,
        minimum_boundary_document_support=args.minimum_boundary_document_support,
    )
    manifest = run_phase1_entity_ablations(config)
    print(
        json.dumps(
            {
                "run_dir": manifest["run_dir"],
                "run_hash": manifest["run_hash"],
                "holdout_status": manifest["holdout_status"],
                "boundary_rule_count": manifest["boundary_rule_audit"]["compiled_rule_count"],
                "variants": [
                    {
                        "name": row["name"],
                        "decision": row["decision"],
                        "all_wer": row["splits"]["all"]["micro_wer_proxy"],
                        "holdout_wer": row["splits"]["holdout"]["micro_wer_proxy"],
                        "holdout_wer_reduction": row["split_deltas"]["holdout"][
                            "wer_reduction"
                        ],
                    }
                    for row in manifest["variants"]
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _parse_stage(value: str, parser: argparse.ArgumentParser) -> tuple[str, Path]:
    if "=" not in value:
        parser.error(f"Invalid --stage {value!r}; expected NAME=PATH.")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path)
    if not name or not path.exists():
        parser.error(f"Invalid source stage {value!r}.")
    return name, path


if __name__ == "__main__":
    main()
