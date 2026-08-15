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
import math
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
    agreement: Mapping[str, Any] | None = None

    @property
    def eligible_for_clinical_claim(self) -> bool:
        """Return whether the manifest and files meet the public clinical-evidence gate."""

        return not self.issues and all(self.checks.values())

    @property
    def eligible_for_engineering_use(self) -> bool:
        """Return whether the dataset is safe for reproducible engineering workflows.

        Synthetic and review-pending fixtures may pass structural, licensing, and split checks
        without being clinical evidence. Keep this gate separate so development benchmarks can
        be used without weakening the clinical-claim contract.
        """

        return not self.issues and all(
            value for key, value in self.checks.items() if key != "human_reviewed_release"
        )

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
            "review_agreement": (
                None
                if self.agreement is None
                else {key: self.agreement[key] for key in sorted(self.agreement)}
            ),
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "eligible_for_engineering_use": self.eligible_for_engineering_use,
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
        rows = _read_rows(
            path,
            split_name,
            issues,
            entity_types=_declared_values(manifest, "entities"),
            assertions=_declared_values(manifest, "assertions"),
            code_systems=_declared_values(manifest, "code_systems"),
        )
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
    agreement, agreement_issues = _load_agreement_report(
        root,
        dataset,
        review,
        human_reviewed=human_reviewed,
    )
    issues.extend(agreement_issues)
    review_fields_ok = _review_contract_ok(
        review,
        human_reviewed,
        reviewed_counts,
        split_counts,
        agreement,
    )
    if human_reviewed and not review_fields_ok:
        issues.append("incomplete_review_contract")

    if status not in _REVIEWED_STATUSES or not human_reviewed:
        warnings.append("clinical_claim_requires_human_review")

    agreement_issue_present = any(
        item.startswith(("missing_agreement_report", "invalid_agreement_report", "agreement_"))
        for item in issues
    )
    annotation_issue_present = any(
        item.startswith(
            (
                "missing_entities:",
                "missing_relations:",
                "invalid_entity:",
                "invalid_relation:",
                "duplicate_entity_id:",
                "duplicate_relation_id:",
            )
        )
        for item in issues
    )
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
        "review_agreement_meets_targets": not agreement_issue_present if human_reviewed else True,
        "annotation_structure_valid": not annotation_issue_present,
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
        agreement=agreement,
    )


def _read_rows(
    path: Path,
    split: str,
    issues: list[str],
    *,
    entity_types: frozenset[str],
    assertions: frozenset[str],
    code_systems: frozenset[str],
) -> list[dict[str, Any]]:
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
        _validate_row_annotations(
            row,
            split,
            line_number,
            issues,
            entity_types=entity_types,
            assertions=assertions,
            code_systems=code_systems,
        )
        rows.append(row)
    return rows


def _validate_row_annotations(
    row: Mapping[str, Any],
    split: str,
    line_number: int,
    issues: list[str],
    *,
    entity_types: frozenset[str],
    assertions: frozenset[str],
    code_systems: frozenset[str],
) -> None:
    """Validate neutral annotation structure without emitting source text in diagnostics."""

    entities = row.get("entities")
    relations = row.get("relations")
    label = f"{split}:{line_number}"
    if not isinstance(entities, list):
        issues.append(f"missing_entities:{label}")
        entities = []
    if not isinstance(relations, list):
        issues.append(f"missing_relations:{label}")
        relations = []
    entity_ids: set[str] = set()
    for entity in entities:
        if not isinstance(entity, Mapping):
            issues.append(f"invalid_entity:{label}")
            continue
        entity_id = entity.get("id")
        if not isinstance(entity_id, str) or not entity_id.strip():
            issues.append(f"invalid_entity:{label}")
        elif entity_id in entity_ids:
            issues.append(f"duplicate_entity_id:{label}")
        else:
            entity_ids.add(entity_id)
        span = entity.get("span")
        start, end = _span_or_none(span)
        if start is None or end is None or not 0 <= start < end <= len(row["text"]):
            issues.append(f"invalid_entity:{label}")
        elif row["text"][start:end] != entity.get("text"):
            issues.append(f"invalid_entity:{label}")
        if entity.get("type") not in entity_types:
            issues.append(f"invalid_entity:{label}")
        if entity.get("assertion") not in assertions:
            issues.append(f"invalid_entity:{label}")
        code_system = entity.get("code_system")
        code = entity.get("code")
        if code_system not in code_systems:
            issues.append(f"invalid_entity:{label}")
        elif code_system == "NONE" and code is not None:
            issues.append(f"invalid_entity:{label}")
        elif code_system != "NONE" and (not isinstance(code, str) or not code.strip()):
            issues.append(f"invalid_entity:{label}")

    relation_ids: set[str] = set()
    for relation in relations:
        if not isinstance(relation, Mapping):
            issues.append(f"invalid_relation:{label}")
            continue
        relation_id = relation.get("id")
        head = relation.get("head")
        tail = relation.get("tail")
        relation_type = relation.get("type")
        if not isinstance(relation_id, str) or not relation_id.strip():
            issues.append(f"invalid_relation:{label}")
        elif relation_id in relation_ids:
            issues.append(f"duplicate_relation_id:{label}")
        else:
            relation_ids.add(relation_id)
        if (
            not isinstance(head, str)
            or not isinstance(tail, str)
            or head == tail
            or head not in entity_ids
            or tail not in entity_ids
            or not isinstance(relation_type, str)
            or not relation_type.strip()
        ):
            issues.append(f"invalid_relation:{label}")


def _span_or_none(value: object) -> tuple[int, int] | tuple[None, None]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    start, end = value
    if isinstance(start, bool) or not isinstance(start, int):
        return None, None
    if isinstance(end, bool) or not isinstance(end, int):
        return None, None
    return start, end


def _declared_values(manifest: Mapping[str, Any], key: str) -> frozenset[str]:
    value = manifest.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        return frozenset()
    return frozenset(value)


def _review_contract_ok(
    review: object,
    human_reviewed: bool,
    reviewed_counts: Mapping[str, int],
    split_counts: Mapping[str, int],
    agreement: Mapping[str, Any] | None,
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
    if any(not _unit_interval(value) for value in agreement.values()):
        return False
    if not all(reviewed_counts.get(split, 0) == count for split, count in split_counts.items()):
        return False
    return agreement is not None


def _load_agreement_report(
    root: Path,
    dataset: Mapping[str, Any],
    review: object,
    *,
    human_reviewed: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load the signed-off agreement evidence required for a public gold release.

    A boolean ``human_reviewed`` flag is not evidence by itself.  The report is a separate,
    hashed artifact so reviewers can reproduce the measured agreement without publishing
    reviewer identities or raw clinical text.
    """

    if not human_reviewed:
        return None, []
    if not isinstance(review, Mapping):
        return None, ["missing_agreement_report"]
    raw_report_path = review.get("agreement_report")
    report_sha = review.get("agreement_report_sha256")
    issues: list[str] = []
    if not isinstance(raw_report_path, str) or not raw_report_path.strip():
        issues.append("missing_agreement_report")
        return None, issues
    if not isinstance(report_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", report_sha):
        issues.append("invalid_agreement_report_sha256")
        return None, issues
    report_path = _safe_child(root, raw_report_path, issues, "agreement_report")
    if report_path is None or not report_path.is_file():
        issues.append("missing_agreement_report")
        return None, issues
    actual_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if actual_sha != report_sha:
        issues.append("agreement_report_fingerprint_mismatch")
        return None, issues
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        issues.append("invalid_agreement_report")
        return None, issues
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "clingrounder.review-agreement.v1":
        issues.append("invalid_agreement_report")
        return None, issues
    if payload.get("dataset_id") != dataset.get("id") or payload.get("dataset_version") != dataset.get("version"):
        issues.append("agreement_dataset_mismatch")
    reviewed_count = payload.get("reviewed_document_count")
    double_reviewed_count = payload.get("double_reviewed_document_count")
    for key in ("reviewed_document_count", "double_reviewed_document_count"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            issues.append(f"agreement_invalid:{key}")
    if (
        isinstance(reviewed_count, int)
        and not isinstance(reviewed_count, bool)
        and isinstance(double_reviewed_count, int)
        and not isinstance(double_reviewed_count, bool)
    ):
        if double_reviewed_count > reviewed_count:
            issues.append("agreement_invalid:double_reviewed_document_count")
    for key in ("double_review_fraction", "span_type_agreement", "assertion_agreement", "relation_agreement"):
        value = payload.get(key)
        if value is not None and not _unit_interval(value):
            issues.append(f"agreement_invalid:{key}")
    fraction = payload.get("double_review_fraction")
    if (
        isinstance(reviewed_count, int)
        and not isinstance(reviewed_count, bool)
        and isinstance(double_reviewed_count, int)
        and not isinstance(double_reviewed_count, bool)
        and _unit_value(fraction) is not None
        and not math.isclose(
            _unit_value(fraction) or 0.0,
            double_reviewed_count / reviewed_count,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        issues.append("agreement_inconsistent:double_review_fraction")
    targets = review.get("agreement_targets")
    if not isinstance(targets, Mapping):
        issues.append("missing_agreement_targets")
    else:
        for key, target in targets.items():
            value = payload.get(f"{key}_agreement")
            target_value = _unit_value(target)
            value_score = _unit_value(value)
            if target_value is None:
                issues.append(f"agreement_invalid_target:{key}")
            elif value_score is None or value_score < target_value:
                issues.append(f"agreement_below_target:{key}")
    target_fraction = review.get("double_review_fraction")
    fraction = payload.get("double_review_fraction")
    target_fraction_value = _unit_value(target_fraction)
    fraction_value = _unit_value(fraction)
    if target_fraction_value is not None:
        if fraction_value is None or fraction_value < target_fraction_value:
            issues.append("agreement_below_target:double_review_fraction")
    return dict(payload), issues


def _unit_interval(value: object) -> bool:
    """Return whether a reported metric is a finite, non-boolean probability."""

    return _unit_value(value) is not None


def _unit_value(value: object) -> float | None:
    """Return a finite [0, 1] metric as a float, or ``None`` for invalid input."""

    return (
        float(value)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        )
        else None
    )


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
