#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.evaluation.entity_wer_report import (
    build_entity_wer_report,
    write_entity_wer_report,
)
from medical_kg_nlp.evaluation.manual_gold import load_phase1_directory
from medical_kg_nlp.evaluation.phase1_ensemble import load_phase1_output_source
from medical_kg_nlp.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase 1 entity WER, source lineage, and boundaries.")
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument("--pred", required=True, help="Final Phase 1 prediction directory or ZIP.")
    parser.add_argument("--documents", default="data/raw/input")
    parser.add_argument(
        "--policy",
        default="data/manual_gold/compiled/phase1_annotation_policy.yaml",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Ordered lineage stage. Repeat from earliest to latest.",
    )
    parser.add_argument("--public-wer", type=float)
    parser.add_argument("--final-source-name", default="final_only")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stages = [_parse_stage(value) for value in args.stage]
    policy_path = Path(args.policy)
    policy = read_yaml(policy_path) if policy_path.exists() else {}
    documents = {
        path.stem: path.read_text(encoding="utf-8") for path in Path(args.documents).glob("*.txt")
    }
    report = build_entity_wer_report(
        gold_by_doc=load_phase1_directory(args.gold_dir),
        pred_by_doc=load_phase1_output_source(args.pred),
        documents_by_doc=documents,
        stages=stages,
        annotation_policy=policy,
        public_wer=args.public_wer,
        final_source_name=args.final_source_name,
    )
    write_entity_wer_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "summary": report["summary"],
                "stage_comparison": report["stage_comparison"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _parse_stage(value: str) -> tuple[str, dict[str, list[dict[str, object]]]]:
    if "=" not in value:
        raise SystemExit(f"Invalid --stage {value!r}; expected NAME=PATH.")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise SystemExit(f"Invalid --stage {value!r}; expected NAME=PATH.")
    return name.strip(), load_phase1_output_source(path.strip())


if __name__ == "__main__":
    main()
