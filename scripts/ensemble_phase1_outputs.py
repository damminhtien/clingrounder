#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.benchmarks.phase1.manual_gold import evaluate_manual_gold, load_phase1_directory
from medical_kg_nlp.benchmarks.phase1.phase1 import (
    validate_phase1_submission_dir,
    validate_phase1_submission_zip,
    zip_phase1_output_dir,
)
from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import (
    PHASE1_ENTITY_TYPE_ORDER,
    Phase1EnsembleSource,
    expand_repeated_phase1_mentions,
    load_phase1_output_source,
    merge_phase1_outputs,
    rank_phase1_source_strategies,
)
from medical_kg_nlp.utils.io import read_source_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a type-wise ensemble from two Phase 1 output directories.")
    parser.add_argument("--primary-dir", required=True, help="Primary Phase 1 directory or ZIP.")
    parser.add_argument("--secondary-dir", required=True, help="Secondary Phase 1 directory or ZIP.")
    parser.add_argument("--input-dir", default="data/raw/input")
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="TYPE=SOURCE",
        help="Explicit source per type; run with --help for supported SOURCE values.",
    )
    parser.add_argument(
        "--search-choice",
        action="append",
        choices=(
            "primary",
            "secondary",
            "union",
            "intersection",
            "primary_preferred_union",
            "secondary_preferred_union",
        ),
        default=[],
        help="Choice included in automatic train-split search; default: primary and secondary.",
    )
    parser.add_argument(
        "--expand-secondary-repeats",
        action="store_true",
        help="Recover all exact same-line occurrences hidden by secondary first-match post-processing.",
    )
    args = parser.parse_args()

    primary = load_phase1_output_source(args.primary_dir)
    secondary = load_phase1_output_source(args.secondary_dir)
    if args.expand_secondary_repeats:
        source_text_by_doc = {
            path.stem: read_source_text(path) for path in Path(args.input_dir).glob("*.txt")
        }
        secondary = expand_repeated_phase1_mentions(secondary, source_text_by_doc)
    gold = load_phase1_directory(args.gold_dir)
    explicit_sources = _parse_sources(args.source)
    search: list[dict[str, Any]] = []
    if explicit_sources:
        source_by_type = explicit_sources
    else:
        choices = tuple(args.search_choice or ["primary", "secondary"])
        search = rank_phase1_source_strategies(
            gold,
            primary,
            secondary,
            choices=cast(tuple[Phase1EnsembleSource, ...], choices),
        )
        source_by_type = cast(dict[str, Phase1EnsembleSource], search[0]["source_by_type"])

    input_dir = Path(args.input_dir)
    document_ids = sorted(
        (path.stem for path in input_dir.glob("*.txt")),
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    predictions = merge_phase1_outputs(
        primary,
        secondary,
        source_by_type,
        document_ids=document_ids,
    )
    output_root = Path(args.output_dir)
    entity_dir = output_root / "output"
    entity_dir.mkdir(parents=True, exist_ok=True)
    for old_path in entity_dir.glob("*.json"):
        old_path.unlink()
    for document_id, rows in predictions.items():
        (entity_dir / f"{document_id}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    validation_issues = [issue.to_json() for issue in validate_phase1_submission_dir(input_dir, entity_dir)]
    if validation_issues:
        raise SystemExit(json.dumps({"validation_issues": validation_issues}, ensure_ascii=False, indent=2))
    zip_path = output_root / "output.zip"
    zip_phase1_output_dir(entity_dir, zip_path)
    zip_issues = [
        issue.to_json()
        for issue in validate_phase1_submission_zip(
            zip_path,
            input_dir=input_dir,
            expected_count=len(document_ids),
        )
    ]
    if zip_issues:
        raise SystemExit(json.dumps({"zip_validation_issues": zip_issues}, ensure_ascii=False, indent=2))

    report = evaluate_manual_gold(gold, predictions)
    summary = {
        "source_by_type": source_by_type,
        "metrics": {name: row["metrics"] for name, row in report["splits"].items()},
        "error_counts": {name: row["error_counts"] for name, row in report["splits"].items()},
        "entity_count": sum(len(rows) for rows in predictions.values()),
        "validation_issue_count": 0,
        "zip": str(zip_path),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "ensemble_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if search:
        (output_root / "strategy_search.json").write_text(
            json.dumps(search, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _parse_sources(values: list[str]) -> dict[str, Phase1EnsembleSource]:
    if not values:
        return {}
    parsed: dict[str, Phase1EnsembleSource] = {}
    allowed_sources = {
        "primary",
        "secondary",
        "union",
        "intersection",
        "primary_preferred_union",
        "secondary_preferred_union",
    }
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --source {value!r}; expected TYPE=SOURCE.")
        entity_type, source = value.split("=", 1)
        if entity_type not in PHASE1_ENTITY_TYPE_ORDER:
            raise SystemExit(f"Unknown Phase 1 entity type: {entity_type}")
        if source not in allowed_sources:
            raise SystemExit(f"Unknown ensemble source: {source}")
        parsed[entity_type] = cast(Phase1EnsembleSource, source)
    missing = set(PHASE1_ENTITY_TYPE_ORDER) - set(parsed)
    if missing:
        raise SystemExit(f"Explicit --source requires all entity types; missing: {sorted(missing)}")
    return parsed


if __name__ == "__main__":
    main()
