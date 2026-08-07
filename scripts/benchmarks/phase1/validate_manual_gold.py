#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.benchmarks.phase1.manual_gold_manifest import validate_manual_gold_manifest
from clingrounder.benchmarks.phase1.phase1 import validate_phase1_entities
from clingrounder.utils.io import read_source_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate manually reviewed Phase 1 gold JSON files against raw TXT offsets and dictionary codes.",
    )
    parser.add_argument("--input-dir", default="data/raw/input", help="Directory containing source TXT files.")
    parser.add_argument("--gold-dir", default="data/manual_gold", help="Directory containing reviewed gold JSON files.")
    parser.add_argument(
        "--dictionary",
        default="data/manual_gold/reviewed_candidate_concepts.jsonl",
        help=(
            "Compact standards-backed dictionary generated from reviewed manual-gold codes. "
            "Rebuild it after changing candidate labels."
        ),
    )
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument(
        "--manifest",
        default="data/manual_gold/review_manifest.jsonl",
        help="Review manifest whose coverage and entity counts must match the gold files.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Validate existing gold files and report missing files without failing.",
    )
    args = parser.parse_args()

    dictionary = DictionaryStore.from_jsonl(args.dictionary)
    input_path = Path(args.input_dir)
    gold_path = Path(args.gold_dir)
    issues: list[dict[str, Any]] = []
    reviewed_files: list[str] = []
    gold_by_id: dict[str, list[dict[str, Any]]] = {}
    entity_count = 0

    expected_ids = [str(index) for index in range(1, args.expected_count + 1)]
    for document_id in expected_ids:
        txt_path = input_path / f"{document_id}.txt"
        json_path = gold_path / f"{document_id}.json"
        if not txt_path.exists():
            issues.append(_issue("missing_input", str(txt_path), f"Missing input TXT for document {document_id}.", document_id))
            continue
        if not json_path.exists():
            if not args.allow_incomplete:
                issues.append(
                    _issue("missing_gold", str(json_path), f"Missing manual gold file for document {document_id}.", document_id)
                )
            continue
        reviewed_files.append(json_path.name)
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            issues.append(_issue("gold_schema", str(json_path), str(error), document_id))
            continue
        if isinstance(payload, list):
            entity_count += len(payload)
            gold_by_id[document_id] = payload
        issues.extend(
            issue.to_json()
            for issue in validate_phase1_entities(
                payload,
                read_source_text(txt_path),
                document_id=document_id,
                dictionary=dictionary,
            )
        )

    extra_files = sorted(
        path.name for path in gold_path.glob("*.json") if path.stem.isdigit() and path.stem not in set(expected_ids)
    )
    for name in extra_files:
        issues.append(_issue("extra_gold", str(gold_path / name), "Manual gold file has no matching expected input id.", Path(name).stem))

    issues.extend(validate_manual_gold_manifest(gold_by_id, args.manifest))

    by_kind: dict[str, int] = {}
    for issue in issues:
        kind = str(issue["kind"])
        by_kind[kind] = by_kind.get(kind, 0) + 1
    summary = {
        "valid": not issues,
        "allow_incomplete": args.allow_incomplete,
        "expected_count": args.expected_count,
        "reviewed_count": len(reviewed_files),
        "missing_count": args.expected_count - len(reviewed_files),
        "entity_count": entity_count,
        "reviewed_files": reviewed_files,
        "issue_count": len(issues),
        "by_kind": dict(sorted(by_kind.items())),
        "issues": issues[:100],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


def _issue(kind: str, path: str, message: str, document_id: str | None) -> dict[str, str]:
    payload = {"kind": kind, "path": path, "message": message}
    if document_id is not None:
        payload["document_id"] = document_id
    return payload


if __name__ == "__main__":
    main()
