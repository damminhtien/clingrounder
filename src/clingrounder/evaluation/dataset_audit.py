"""Audit neutral benchmark datasets before they are used as public evidence.

The audit is deliberately independent of any task-specific scorer.  It checks the properties
that make a dataset reproducible and scientifically interpretable: immutable split files,
non-overlapping templates, review provenance, and source/license declarations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping

import yaml

__all__ = ["DatasetAuditReport", "audit_dataset"]

_MANIFEST_SCHEMA = "clingrounder.dataset-manifest.v1"
_REVIEWED_STATUSES = {"human_reviewed", "reviewed", "released"}


@dataclass(frozen=True, slots=True)
class DatasetAuditReport:
    """Machine-readable evidence about one benchmark dataset directory."""

    dataset_id: str
    dataset_version: str
    status: str
    human_reviewed: bool
    document_count: int
    split_counts: Mapping[str, int]
    split_fingerprints: Mapping[str, str]
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    checks: Mapping[str, bool]

    @property
    def eligible_for_clinical_claim(self) -> bool:
        """Return whether the manifest and files meet the public clinical-evidence gate."""

        return not self.issues and all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        """Render stable JSON without exposing document text or mention strings."""

        return {
            "schema_version": "clingrounder.dataset-audit.v1",
            "dataset": {
                "id": self.dataset_id,
                "version": self.dataset_version,
                "status": self.status,
                "human_reviewed": self.human_reviewed,
            },
            "documents": {
                "total": self.document_count,
                "by_split": dict(sorted(self.split_counts.items())),
            },
            "split_fingerprints": dict(sorted(self.split_fingerprints.items())),
            "checks": dict(sorted(self.checks.items())),
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "eligible_for_clinical_claim": self.eligible_for_clinical_claim,
        }


def audit_dataset(benchmark_dir: str | Path) -> DatasetAuditReport:
    """Audit a manifest and all declared JSONL splits.

    The function is deterministic and returns a report instead of silently promoting a dataset.
    It hashes text for leakage checks but never writes the text or mention content into the
    report.  A synthetic or review-pending dataset is valid for engineering tests, but it cannot
    pass ``eligible_for_clinical_claim``.
    """

    root = Path(benchmark_dir).expanduser().resolve()
    manifest_path = root / "dataset_manifest.yaml"
    issues: list[str] = []
    warnings: list[str] = []
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read dataset manifest: {manifest_path}") from error
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != _MANIFEST_SCHEMA:
        raise ValueError(f"unsupported dataset manifest: {manifest_path}")

    dataset = _mapping(manifest, "dataset")
    dataset_id = _required_string(dataset, "id")
    dataset_version = _required_string(dataset, "version")
    status = _required_string(dataset, "status")
    human_reviewed = dataset.get("human_reviewed") is True
    if not _required_string(dataset, "license"):
        issues.append("missing_license")
    if not _required_string(dataset, "license_url"):
        issues.append("missing_license_url")

    splits = _mapping(manifest, "splits")
    split_counts: dict[str, int] = {}
    split_fingerprints: dict[str, str] = {}
    template_groups: dict[str, set[str]] = defaultdict(set)
    text_hashes: dict[str, set[str]] = defaultdict(set)
    seen_document_ids: dict[str, str] = {}
    reviewed_counts: dict[str, int] = {}
    declared_count_ok = True
    declared_hash_ok = True

    for split_name, raw_split in sorted(splits.items()):
        if not isinstance(split_name, str) or not split_name.strip():
            issues.append("empty_split_name")
            continue
        split = _mapping(splits, split_name)
        relative_path = _required_string(split, "path")
        path = _safe_child(root, relative_path, issues, f"split_path:{split_name}")
        if path is None or not path.is_file():
            issues.append(f"missing_split:{split_name}")
            continue
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        split_fingerprints[split_name] = actual_sha
        declared_sha = split.get("sha256")
        if declared_sha is not None and declared_sha != actual_sha:
            declared_hash_ok = False
            issues.append(f"fingerprint_mismatch:{split_name}")
        rows = _read_rows(path, split_name, issues)
        split_counts[split_name] = len(rows)
        declared_count = split.get("documents")
        if declared_count is not None and declared_count != len(rows):
            declared_count_ok = False
            issues.append(f"document_count_mismatch:{split_name}")
        for row in rows:
            document_id = row["document_id"]
            previous_split = seen_document_ids.get(document_id)
            if previous_split is not None:
                issues.append(f"duplicate_document_id:{document_id}")
            else:
                seen_document_ids[document_id] = split_name
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                issues.append(f"invalid_metadata:{split_name}:{document_id}")
                continue
            template = metadata.get("template_group")
            if isinstance(template, str) and template:
                template_groups[split_name].add(template)
            normalized_hash = _text_fingerprint(row["text"])
            text_hashes[split_name].add(normalized_hash)
            if _is_reviewed(metadata.get("human_reviewed")):
                reviewed_counts[split_name] = reviewed_counts.get(split_name, 0) + 1

    overlap_templates = _cross_split_overlap(template_groups)
    if overlap_templates:
        issues.append("template_group_overlap")
    overlap_text = _cross_split_overlap(text_hashes)
    if overlap_text:
        issues.append("normalized_text_overlap")

    policy = manifest.get("policy")
    if not isinstance(policy, Mapping):
        issues.append("missing_policy")
        policy = {}
    if policy.get("test_used_for_development") is not False:
        issues.append("test_development_policy_missing")
    if policy.get("private_data") is not False:
        issues.append("private_data_policy_missing")
    if policy.get("template_groups_disjoint") is True and overlap_templates:
        issues.append("declared_disjoint_templates_overlap")

    review = manifest.get("review")
    review_fields_ok = _review_contract_ok(review, human_reviewed, reviewed_counts, split_counts)
    if human_reviewed and not review_fields_ok:
        issues.append("incomplete_review_contract")

    if status not in _REVIEWED_STATUSES or not human_reviewed:
        warnings.append("clinical_claim_requires_human_review")

    checks = {
        "declared_split_counts_match": declared_count_ok,
        "declared_split_fingerprints_match": declared_hash_ok,
        "document_ids_unique": not any(item.startswith("duplicate_document_id:") for item in issues),
        "template_groups_disjoint": not overlap_templates,
        "normalized_text_splits_disjoint": not overlap_text,
        "source_license_declared": not any(
            item in {"missing_license", "missing_license_url"} for item in issues
        ),
        "review_contract_complete": review_fields_ok,
        "human_reviewed_release": status in _REVIEWED_STATUSES and human_reviewed,
        "test_not_used_for_development": policy.get("test_used_for_development") is False,
    }
    return DatasetAuditReport(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        status=status,
        human_reviewed=human_reviewed,
        document_count=sum(split_counts.values()),
        split_counts=split_counts,
        split_fingerprints=split_fingerprints,
        issues=tuple(sorted(set(issues))),
        warnings=tuple(sorted(set(warnings))),
        checks=checks,
    )


def _read_rows(path: Path, split: str, issues: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        issues.append(f"unreadable_split:{split}")
        return rows
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            issues.append(f"invalid_json:{split}:{line_number}")
            continue
        if not isinstance(row, dict) or not isinstance(row.get("document_id"), str):
            issues.append(f"invalid_document:{split}:{line_number}")
            continue
        if not isinstance(row.get("text"), str) or not row["text"]:
            issues.append(f"empty_text:{split}:{line_number}")
            continue
        rows.append(row)
    return rows


def _review_contract_ok(
    review: object,
    human_reviewed: bool,
    reviewed_counts: Mapping[str, int],
    split_counts: Mapping[str, int],
) -> bool:
    if not human_reviewed:
        return True
    if not isinstance(review, Mapping):
        return False
    if review.get("reviewers_required", 0) < 2:
        return False
    double_review_fraction = review.get("double_review_fraction")
    if not isinstance(double_review_fraction, (int, float)) or not 0.1 <= double_review_fraction <= 1.0:
        return False
    agreement = review.get("agreement_targets")
    if not isinstance(agreement, Mapping):
        return False
    if any(
        not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0
        for value in agreement.values()
    ):
        return False
    return all(reviewed_counts.get(split, 0) == count for split, count in split_counts.items())


def _cross_split_overlap(values: Mapping[str, set[str]]) -> tuple[str, ...]:
    seen: set[str] = set()
    overlap: set[str] = set()
    for split in sorted(values):
        overlap.update(seen.intersection(values[split]))
        seen.update(values[split])
    return tuple(sorted(overlap))


def _text_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_child(root: Path, relative_path: str, issues: list[str], label: str) -> Path | None:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        issues.append(f"path_traversal:{label}")
        return None
    return path


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _is_reviewed(value: object) -> bool:
    return value is True or (isinstance(value, str) and value.casefold() == "true")
