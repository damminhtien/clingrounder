#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.benchmarks.phase1.manual_gold import (
    evaluate_manual_gold,
    load_phase1_directory,
    write_manual_gold_report,
)
from medical_kg_nlp.benchmarks.phase1.phase1 import (
    validate_phase1_submission_dir,
    validate_phase1_submission_zip,
    zip_phase1_output_dir,
)
from medical_kg_nlp.benchmarks.phase1.phase1_candidate_overlay import (
    Phase1CandidateIndex,
    Phase1CandidateOverlayConfig,
    apply_phase1_candidate_overlay,
    candidate_ablation_passes,
)
from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import load_phase1_output_source


_VARIANTS = (
    ("C0_EMPTY", Phase1CandidateOverlayConfig()),
    ("C1_ICD_EXACT", Phase1CandidateOverlayConfig(icd_exact=True)),
    (
        "C2_RXNORM_EXACT",
        Phase1CandidateOverlayConfig(icd_exact=True, rxnorm_exact=True),
    ),
    (
        "C3_RXNORM_LONGEST",
        Phase1CandidateOverlayConfig(
            icd_exact=True,
            rxnorm_exact=True,
            rxnorm_longest=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict Phase 1 candidate overlay ablations.")
    parser.add_argument("--base", required=True, help="Entity output directory or ZIP.")
    parser.add_argument(
        "--dictionary",
        default="data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl",
    )
    parser.add_argument("--gold-dir", default="data/manual_gold")
    parser.add_argument("--input-dir", default="data/raw/input")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base = load_phase1_output_source(args.base)
    gold = load_phase1_directory(args.gold_dir)
    index = Phase1CandidateIndex.from_jsonl(args.dictionary)
    dictionary = DictionaryStore.from_jsonl(args.dictionary)
    output_root = Path(args.output_dir)
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)
    expected_count = len(list(input_dir.glob("*.txt")))

    results: list[dict[str, Any]] = []
    accepted_name = "C0_EMPTY"
    accepted_holdout: dict[str, Any] | None = None
    for name, config in _VARIANTS:
        predictions, assignment_counts = apply_phase1_candidate_overlay(base, index, config)
        report = evaluate_manual_gold(gold, predictions)
        variant_root = output_root / name
        write_manual_gold_report(report, variant_root / "evaluation")
        holdout_metrics = report["splits"]["holdout"]["metrics"]
        passed = accepted_holdout is None or candidate_ablation_passes(accepted_holdout, holdout_metrics)
        if passed:
            accepted_name = name
            accepted_holdout = holdout_metrics

        entity_dir = variant_root / "output"
        _write_rows(predictions, entity_dir)
        issues = validate_phase1_submission_dir(input_dir, entity_dir, dictionary=dictionary)
        if issues:
            raise SystemExit(_issues_json(name, issues))
        zip_path = variant_root / "output.zip"
        zip_phase1_output_dir(entity_dir, zip_path)
        zip_issues = validate_phase1_submission_zip(
            zip_path,
            input_dir=input_dir,
            dictionary=dictionary,
            expected_count=expected_count,
        )
        if zip_issues:
            raise SystemExit(_issues_json(name, zip_issues))
        results.append(
            {
                "name": name,
                "config": {
                    "icd_exact": config.icd_exact,
                    "rxnorm_exact": config.rxnorm_exact,
                    "rxnorm_longest": config.rxnorm_longest,
                },
                "assignment_counts": assignment_counts,
                "accepted": passed,
                "metrics": {
                    split: split_report["metrics"]
                    for split, split_report in report["splits"].items()
                },
                "error_counts": {
                    split: split_report["error_counts"]
                    for split, split_report in report["splits"].items()
                },
                "selective_prediction": {
                    split: split_report["selective_prediction"]["candidates"]
                    for split, split_report in report["splits"].items()
                },
                "zip": str(zip_path),
            }
        )

    final_root = output_root / "final"
    shutil.copytree(output_root / accepted_name, final_root)
    payload = {
        "accepted_variant": accepted_name,
        "selection_scope": "local_pre_submit_only",
        "promotion_status": "requires_public_validation",
        "sources": {
            "base": {"path": args.base, "sha256": _path_sha256(Path(args.base))},
            "dictionary": {
                "path": args.dictionary,
                "sha256": _path_sha256(Path(args.dictionary)),
            },
        },
        "gate": "local holdout score and candidates_score must both increase",
        "variants": results,
        "final_zip": str(final_root / "output.zip"),
    }
    (output_root / "ablation_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.md").write_text(_summary_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _write_rows(rows_by_doc: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("*.json"):
        old_path.unlink()
    for document_id, rows in rows_by_doc.items():
        (output_dir / f"{document_id}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _issues_json(name: str, issues: list[Any]) -> str:
    return json.dumps(
        {"variant": name, "validation_issues": [issue.to_json() for issue in issues]},
        ensure_ascii=False,
        indent=2,
    )


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 1 Candidate Ablation",
        "",
        "| Variant | Assigned | Train | Holdout | Holdout candidate | Accepted |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for variant in payload["variants"]:
        metrics = variant["metrics"]
        lines.append(
            f"| {variant['name']} | {variant['assignment_counts'].get('assigned_total', 0)} "
            f"| {metrics['train']['score']:.4f} | {metrics['holdout']['score']:.4f} "
            f"| {metrics['holdout']['candidates_score']:.6f} | {variant['accepted']} |"
        )
    lines.extend(
        [
            "",
            f"Local selection: `{payload['accepted_variant']}`",
            "",
            "Public promotion still requires an external grader result.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
