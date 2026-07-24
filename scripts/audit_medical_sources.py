#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_kg_nlp.dictionaries.source_audit import build_source_audit_report, write_source_audit_report

_DEFAULT_LOCAL_FILES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "icd10_vn_tt06_2026",
        "role": "raw_pdf",
        "path": "data/standards/icd10_vn/raw/06-byt-kem.pdf",
        "required": True,
    },
    {
        "source_id": "icd10_vn_tt06_2026",
        "role": "processed_extract_jsonl",
        "path": "data/standards/icd10_vn/processed/tt06_icd10_extract.jsonl",
        "required": True,
    },
    {
        "source_id": "icd10_vn_tt06_2026",
        "role": "processed_concepts_jsonl",
        "path": "data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl",
        "required": True,
    },
    {
        "source_id": "rxnorm_prescribable_2026_07_06",
        "role": "raw_zip",
        "path": "data/standards/rxnorm/raw/RxNorm_full_07062026.zip",
        "required": True,
    },
    {
        "source_id": "rxnorm_prescribable_2026_07_06",
        "role": "processed_concepts_jsonl",
        "path": "data/standards/rxnorm/processed/rxnorm_prescribable_07062026_concepts.jsonl",
        "required": True,
    },
    {
        "source_id": "rxnorm_full_2026_07_06",
        "role": "processed_concepts_jsonl",
        "path": "data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl",
        "required": True,
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit source registry, raw checksums, dictionary coverage, and terminology QA gaps.",
    )
    parser.add_argument("--registry", default="data/sources/medical_resource_registry.yaml")
    parser.add_argument("--standard-versions", default="data/standards/source_versions.json")
    parser.add_argument(
        "--dictionary",
        action="append",
        default=[],
        help="Dictionary ConceptEntry JSONL to profile. Repeatable.",
    )
    parser.add_argument(
        "--raw-file",
        action="append",
        default=[],
        help="Extra local file as source_id:role:path[:required|optional]. Repeatable.",
    )
    parser.add_argument(
        "--rxnorm-release",
        action="append",
        default=[],
        help="RxNorm ZIP/RRF to profile as path[::archive/member/root[::prescribable|full]].",
    )
    parser.add_argument("--input-dir", help="Optional TXT folder for unknown mention candidate mining.")
    parser.add_argument("--output-dir", default="outputs/source_audit/current")
    parser.add_argument("--unknown-top-k", type=int, default=100)
    args = parser.parse_args()

    dictionaries = args.dictionary or [
        "data/dictionaries/seed_concepts.jsonl",
        "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl",
    ]
    rxnorm_releases = (
        [_parse_rxnorm_release(item) for item in args.rxnorm_release]
        if args.rxnorm_release
        else _default_rxnorm_releases()
    )
    local_files = [*_DEFAULT_LOCAL_FILES, *[_parse_raw_file(item) for item in args.raw_file]]
    report = build_source_audit_report(
        registry_path=args.registry,
        standard_versions_path=args.standard_versions,
        dictionary_paths=dictionaries,
        local_files=local_files,
        rxnorm_release_paths=rxnorm_releases,
        input_dir=args.input_dir,
        unknown_top_k=args.unknown_top_k,
    )
    write_source_audit_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "source_manifest": str(Path(args.output_dir) / "source_manifest.json"),
                "dictionary_coverage": str(Path(args.output_dir) / "dictionary_coverage.md"),
                "manual_review_queue": str(Path(args.output_dir) / "manual_review_queue.jsonl"),
                "false_positive_blocklist": str(Path(args.output_dir) / "false_positive_blocklist.jsonl"),
                "summary": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _parse_raw_file(value: str) -> dict[str, Any]:
    parts = value.split(":")
    if len(parts) < 3:
        raise ValueError(f"Expected source_id:role:path[:required|optional], got {value!r}")
    source_id, role = parts[0], parts[1]
    required = True
    if parts[-1] in {"required", "optional"}:
        required = parts[-1] == "required"
        path = ":".join(parts[2:-1])
    else:
        path = ":".join(parts[2:])
    return {"source_id": source_id, "role": role, "path": path, "required": required}


def _parse_rxnorm_release(value: str) -> str | dict[str, str]:
    parts = value.split("::", maxsplit=2)
    if len(parts) == 1:
        return parts[0]
    path, member_root = parts[:2]
    content = parts[2] if len(parts) == 3 else "prescribable"
    if not path or not member_root or content not in {"prescribable", "full"}:
        raise ValueError(f"Expected path::archive/member/root[::prescribable|full], got {value!r}")
    return {"path": path, "archive_member_root": member_root, "content": content}


def _default_rxnorm_releases() -> list[dict[str, str]]:
    july_bundle = Path("data/standards/rxnorm/raw/RxNorm_full_07062026.zip")
    if not july_bundle.exists():
        return []
    return [
        {
            "path": str(july_bundle),
            "archive_member_root": "prescribe/rrf",
            "content": "prescribable",
        },
        {
            "path": str(july_bundle),
            "archive_member_root": "rrf",
            "content": "full",
        },
    ]


if __name__ == "__main__":
    main()
