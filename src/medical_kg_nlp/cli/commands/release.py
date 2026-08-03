"""Public repository release-audit command."""

from __future__ import annotations

import argparse
from pathlib import Path

from medical_kg_nlp.governance.public_release import (
    audit_public_repository,
    report_json,
)

__all__ = ["audit_release"]


def audit_release(args: argparse.Namespace) -> int:
    """Fail when tracked bytes violate the checked-in publication policy."""

    report = audit_public_repository(args.root, args.policy)
    payload = report_json(report) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report.valid else 1
