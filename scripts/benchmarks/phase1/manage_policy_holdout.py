#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from clingrounder.benchmarks.phase1.policy_holdout import (
    build_policy_holdout_manifest,
    open_policy_holdout_manifest,
    verify_policy_holdout_manifest,
    write_policy_holdout_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create, verify, or open a source-first blind policy holdout.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--documents", required=True)
    create.add_argument("--corpus-id", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--modulus", type=int, default=5)
    create.add_argument("--holdout-bucket", type=int, default=0)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--documents", required=True)
    verify.add_argument("--manifest", required=True)

    opened = subparsers.add_parser("open")
    opened.add_argument("--documents", required=True)
    opened.add_argument("--manifest", required=True)
    opened.add_argument("--holdout-gold-dir", required=True)
    opened.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.action == "create":
        manifest = build_policy_holdout_manifest(
            args.documents,
            corpus_id=args.corpus_id,
            modulus=args.modulus,
            holdout_bucket=args.holdout_bucket,
        )
        write_policy_holdout_manifest(manifest, args.output)
    else:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if args.action == "verify":
            verify_policy_holdout_manifest(manifest, args.documents)
        else:
            manifest = open_policy_holdout_manifest(
                manifest,
                args.documents,
                args.holdout_gold_dir,
            )
            write_policy_holdout_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "action": args.action,
                "status": manifest["status"],
                "corpus_id": manifest["corpus_id"],
                "source_fingerprint_sha256": manifest["corpus"][
                    "source_fingerprint_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
