#!/usr/bin/env python
"""Synchronize or verify pinned external architecture references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clingrounder.experiments.reference_implementations import (
    load_reference_registry,
    sync_reference_checkouts,
    verify_reference_checkouts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/references/clinical_nlp_sources.json",
    )
    parser.add_argument(
        "--checkout-root",
        default="external/reference_repos",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Fetch and detach every checkout at its pinned revision before verification.",
    )
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()

    registry = load_reference_registry(args.registry)
    results = (
        sync_reference_checkouts(registry, args.checkout_root)
        if args.sync
        else verify_reference_checkouts(registry, args.checkout_root)
    )
    report = {
        "schema_version": "clinical-nlp-reference-verification.v1",
        "registry_schema_version": registry.schema_version,
        "valid": all(result.valid for result in results),
        "sources": [result.to_json() for result in results],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
