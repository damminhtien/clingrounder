"""Profile-aware prediction and release-artifact validation command."""

from __future__ import annotations

import argparse
import json
import sys

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.schema.validator import PredictionValidationIssue, PredictionValidator
from clingrounder.utils.io import read_jsonl
from clingrounder.terminology.memory import InMemoryTerminologyRepository
from clingrounder.validation import (
    ValidationProfile,
    ValidationSeverity,
    apply_validation_profile,
    validate_artifact,
)

__all__ = ["validate"]


def validate(args: argparse.Namespace) -> int:
    """Validate all rows, classify severity, and fail only on blocking errors."""

    profile = ValidationProfile(args.profile)
    documents = _load_documents(args.documents) if args.documents else {}
    dictionary = DictionaryStore.from_jsonl(args.dictionary) if args.dictionary else None
    terminology = (
        InMemoryTerminologyRepository(dictionary) if dictionary is not None else None
    )
    validator = PredictionValidator(terminology)
    profiled = []
    seen_documents: set[str] = set()
    rows = read_jsonl(args.pred)
    for line_number, row in enumerate(rows, start=1):
        document_id = str(row.get("document_id", ""))
        issues: list[PredictionValidationIssue] = []
        if document_id in seen_documents:
            issues.append(
                PredictionValidationIssue(
                    "duplicate_document_id",
                    "$.document_id",
                    f"Duplicate document id {document_id!r} at line {line_number}.",
                )
            )
        seen_documents.add(document_id)
        _, row_issues = validator.validate_payload(
            row,
            source_text=documents.get(document_id),
        )
        issues.extend(row_issues)
        profiled.extend(
            apply_validation_profile(
                issues,
                profile,
            )
        )

    artifact_issues = (
        validate_artifact(
            args.artifact,
            profile=profile,
            expected_files=tuple(args.expected_file),
        )
        if args.artifact
        else []
    )
    for item in profiled:
        print(json.dumps(item.to_json(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
    for artifact_issue in artifact_issues:
        print(
            json.dumps(artifact_issue.to_json(), ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
    errors = sum(item.severity is ValidationSeverity.ERROR for item in profiled) + len(
        artifact_issues
    )
    warnings = sum(item.severity is ValidationSeverity.WARNING for item in profiled)
    print(
        json.dumps(
            {
                "profile": profile.value,
                "rows": len(rows),
                "errors": errors,
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if errors else 0


def _load_documents(path: str) -> dict[str, str]:
    documents: dict[str, str] = {}
    for row in read_jsonl(path):
        document_id = str(row["document_id"])
        if document_id in documents:
            raise ValueError(f"Duplicate source document id {document_id!r}")
        documents[document_id] = str(row["text"])
    return documents
