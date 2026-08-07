#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from clingrounder.benchmarks.phase1.phase1_probe_suite import (
    Phase1Top10ProbeConfig,
    build_phase1_top10_probe_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build isolated, strict-validated Phase 1 Top 10 probe artifacts."
    )
    parser.add_argument("--base", required=True, help="Frozen baseline output directory or ZIP.")
    parser.add_argument(
        "--expected-base-sha256",
        help="Fail closed unless the frozen baseline has this SHA-256.",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Independent proposal source; repeat at least twice, ideally pipeline/Qwen/Codex.",
    )
    parser.add_argument("--input-dir", default="data/raw/input")
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument("--review-manifest", default="data/manual_gold/review_manifest.jsonl")
    parser.add_argument(
        "--dictionary",
        default="data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl",
    )
    parser.add_argument("--rule-registry", help="Reviewed runtime rule registry; drafts stay inactive.")
    parser.add_argument("--output-root", default="outputs/phase1/top10_probes")
    parser.add_argument("--journal-dir", default="outputs/loops/journal")
    parser.add_argument("--minimum-boundary-document-support", type=int, default=2)
    parser.add_argument(
        "--minimum-candidate-proposal-sources",
        type=int,
        default=3,
        help="Exact agreeing proposal sources required before a candidate can be emitted.",
    )
    parser.add_argument(
        "--open-holdout",
        action="store_true",
        help="Day-6-only switch: include the sealed holdout metrics in reports.",
    )
    parser.add_argument(
        "--full-diagnostic",
        action="store_true",
        help="Build full ICD/RxNorm/combined artifacts for local ceiling analysis.",
    )
    args = parser.parse_args()
    if args.minimum_boundary_document_support < 1:
        parser.error("--minimum-boundary-document-support must be at least 1")
    if args.minimum_candidate_proposal_sources < 2:
        parser.error("--minimum-candidate-proposal-sources must be at least 2")
    sources = _parse_sources(args.source, parser)
    config = Phase1Top10ProbeConfig(
        base=Path(args.base),
        proposal_sources=sources,
        input_dir=Path(args.input_dir),
        gold_dir=Path(args.gold_dir),
        review_manifest=Path(args.review_manifest),
        dictionary=Path(args.dictionary),
        output_root=Path(args.output_root),
        journal_dir=Path(args.journal_dir),
        rule_registry=Path(args.rule_registry) if args.rule_registry else None,
        minimum_boundary_document_support=args.minimum_boundary_document_support,
        minimum_candidate_proposal_sources=args.minimum_candidate_proposal_sources,
        expected_base_sha256=args.expected_base_sha256,
        open_holdout=args.open_holdout,
        full_diagnostic=args.full_diagnostic,
    )
    manifest = build_phase1_top10_probe_suite(config)
    print(
        json.dumps(
            {
                "run_dir": manifest["run_dir"],
                "run_hash": manifest["run_hash"],
                "holdout_status": manifest["holdout_status"],
                "tri_source_ready": manifest["tri_source_ready"],
                "candidate_consensus_ready": manifest["candidate_consensus_ready"],
                "candidate_consensus_key_count": manifest["candidate_consensus_key_count"],
                "probe_ready": [
                    row["name"] for row in manifest["variants"] if row["probe_ready"]
                ],
            },
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
        path = Path(raw_path)
        if not name or name in sources:
            parser.error(f"Duplicate or empty proposal source {name!r}.")
        if not path.exists():
            parser.error(f"Proposal source does not exist: {path}")
        sources[name] = path
    if len(sources) < 2:
        parser.error("At least two --source arguments are required.")
    return sources


if __name__ == "__main__":
    main()
